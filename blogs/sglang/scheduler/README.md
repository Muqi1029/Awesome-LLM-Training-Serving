[Batching](https://www.notion.so/Batching-2008a0af5e7f80a4803aeb796320b234?pvs=21)

## Overview

ScheduleBatch：Store all information of a batch on the scheduler.

maintain the following important states

```python
running_batch: ScheduleBatch(reqs=[]) # running_batch
last_batch: Optional[ScheduleBatch] # last forward batch
cur_batch: Optional[ScheduleBatch] # current forward batch
waiting_queue: List[Req] # waiting queue
```

### Log

`Scheduler Args` :

- `forward_ct=0` : used in the `watchdog`
- `forward_ct_decode=0` : control the log frequency
  - `self.forward_ct_decode *%* self.server_args.decode_log_interval *==* 0`
- `num_generated_tokens=0` : added in the `process_batch_result_decode`
- `num_prefill_tokens=0` : unused
- `last_decode_stats_tic=time.perf_counter()`
- `last_prefill_stats_tic=time.perf_counter()` : unused

```python
def run_batch():
  forward_ct += 1
```

**`Log Prefill:`**

1. num_new_seq: `len(can_run_list)`
2. new-token: `adder.log_input_tokens`
3. cached-token: `adder.log_hit_tokens`
4. token usage
   1. `num_used:`
      1. `unused: token_to_kv_pool_allocator.available_size() + tree_cache.evictable_size()`
      2. `max_total_num_tokens - unused_num_tokens`
   2. `num_used / max_total_num_tokens`
5. #queue-req: `len(self.waiting_queue)`

**`Log decode:`**

1. #running-req: `num_running_reqs`
2. #token: `num_used`
3. token usage: same to above
4. cuda graph: `can_run_cuda_graph`
5. gen throughput (token/s): `self.num_generated_tokens */* gap_latency`
6. #queue-req: `len(self.waiting_queue)`

### Watch Dog

**`Server Args`**:

- `watchdog_timeout`: float ("Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.")

```python
def watchdog_thread():
  self.watchdog_last_forward_ct = 0
  self.watchdog_last_time = time.perf_counter()

  while True:
    current = time.perf_counter()
    if self.cur_batch is not None:
      if self.watchdog_last_forward_ct == self.forward_ct:
        # timeout check
        if current > self.watchdog_last_time + self.watchdog_timeout:
          break
      else:
        self.watchdog_last_forward_ct = self.forward_ct
        self.watchdog_last_time = current
    time.sleep(self.watchdog_timeout // 2)
  # dump the process

  # kill the parent_process: `TokenizerManager`
  self.parent_process.send_signal(signal.SIGQUIT)
```

## Scheduler Event Loop

```python
def event_loop_overlap():
  self.result_queue = queue()
  while True:
    recv_reqs = self.recv_requests()
    self.process_input_requests(recv_reqs)

    batch = self.get_next_batch_to_run()
    self.cur_batch = batch


    result = self.run_batch(batch)
    self.process_batch_result(result)

    self.last_batch = batch
```

1. `recv_requests` : Receive results at `tp_rank = 0` and broadcast it to all other TP ranks.

```python
def recv_requests(self) -> List[Req]:
    """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""
    if self.attn_tp_rank == 0:
        recv_reqs = []
         while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except:
                ...
   else:
     recv_reqs = []

    return recv_reqs
```

1. `process_input_requests` ：将`TokenizedGenerateReqInput` 等封装成`Req` ，在这个过程中进行一些参数检查和记录

   1. `handle_generate_request` : 将各种 Input 转换为`Req`
   2. `_add_request_to_queue` : 根据当前 Scheduler 的类型将`Req` 加入到不同的队列中，其中如果是混合（默认）Scheduler 的话，就是将`Req`加入到`waiting_queue` 中

1. `get_next_batch_to_run` ：获得一个`ScheduleBatch` 对象 (**CORE**)

> Why `prefill` is prioritized?
> TTFT!

```python
def get_new_batch_to_run():
  # update running_batch
  if last_batch.forward_mode == "extend":
    # consider chunk prefill
    ...

    # add prefilled batch into running_batch for next schedule for decode
    running_batch.merge(last_batch)

  # first consider prefilling
  new_batch = get_new_batch_prefill()
  if new_batch is not None:
    return new_batch # let the cur_batch be this prefill batch

  # run decode by filter some finished batch
  running_batch = update_running_batch(running_batch)
  return running_batch
```

`get_new_batch_prefill`:

```python
def get_new_batch_prefill():
  if (running_batch.is_full or len(waitting_queue) == 0) and chunk_req is None:
    return

  running_bs = len(running_batch.reqs)

  # whether the prefix has been computed
  # sort the reqs in the waiting queue by policies
  prefix_computed = policy.calc_priority(waiting_queue)

  addr = PrefillAdder(tree_cache, ...)

  for req in waiting_queue:
    # get prefix_indices & last_node from tree_cache => compute extend_input_len
    # put above info into req obj which is used for whether add it into new_batch
    req.init_next_round_input()

    # check whether the extended token num surpass accountable tokens
    # if not surpass, add this req into addr.can_run_list
    addr.add_one_req(req, ...)


  waiting_queue.remove(addr.can_run_list)

  # create a new `ScheduleBatch`
  new_batch = ScheduleBatch.init_new(can_run_list)

  # allocate resources
  new_batch.prepare_for_extend()

  return new_batch
```

<aside>
💡

Scheduler Policy mainly controls the `get_new_prefill_batch`

`Server Args: schedule_policy=”fcfs”`

```python
# scheduler init process
self.policy = SchedulePolicy(schedule_policy, tree_cache, enable_hi_cache)
```

`Server Args`

```python
class PrefillAdder:
  def __init__(self):

  def add_one_req(req, ...):
    # estimate the total_tokens (to eos_token) used by this req
    total_tokens = req.extend_input_len + min(
            req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION
        )

        # get this num_input_tokens (n * page_size)
        input_tokens = (
            -(-req.extend_input_len // self.tree_cache.page_size)
            * self.tree_cache.page_size
        )

        # get prefix_len tokens in this req
        prefix_len = len(req.prefix_indices)

        if total_tokens >= self.rem_total_tokens:
          # meaning the required total tokens by this req cannot be satified
          return AddReqResult.NO_TOKEN

      if input_tokens > self.rem_input_tokens and len(self.can_run_list) != 0:
            # extend_len tokens > rem_input_tokens and
            return AddReqResult.OTHER

    with self._lock_node(req.last_node):

```

There are totally Policies:

1. `CacheAware` (require prefix matches)
   1. `Longest Prefix`
   2. `DFS-Weight`

```python
# in SechdulePolicy
def _compute_prefix_matches():
  waiting_queue_radix_tree.reset()

  for req in waiting_queue:
    # get req's prefix ids
    prefix_ids = req.adjust_max_prefix_ids()

    # use tree cache to match (rid is not used in matching prefix)
    req.prefix_indices, req.last_node = tree_cache.match_prefix(req.rid, prefix_ids)

```

1. `CacheAgnostic`
   1. `FCFS`
   2. `Longest output first`
   3. `Random`

</aside>

`update_running_batch`:

```python
def update_running_batch(running_batch):
  initial_bs = running_batch.batch_size()

  running_batch.filter_batch()

  # check if decode out of memory

  if running_batch.batch_size() < initial_bs:
    running_batch.batch_is_full = False

  running_batch.prepare_for_decode()
  return running_batch
```

1. `run_batch` ：真正运行 forward 的地方，返回一个`GenerationBatchResult` (临时封装产生一个 token 的结果) 从`ScheduleBatch`中获得`ModelWorkerBatch` ，(`ScheduleBatch`的再高一层的封装）

```python
def run_batch():

 # get model_worker_batch from `ScheduleBatch`
 model_worker_batch = batch.get_model_worker_batch()

 # tp worker to forward
  logits_output, next_token_ids, ... \
      = tp_worker.forward_batch_generation(model_worker_batch)

 batch.output_ids = next_token_ids
 bid = model_worker_batch.bid

 # wrap results
 ret = GenerationBatchResult(logits_output, ... , next_token_ids, bid, ...)
 return ret
```

1. `process_batch_result`

处理的过程中，根据`forward_mode`来选择`process`的方法，将`ScheduleBatch` 与`GenerationBatchResult` 作为参数进行处理对比

给`send_to_detokenizer` 发送 `BatchTokenIDOut`

```python
def process_batch_result(
    self,
    batch: ScheduleBatch,
    result: Union[GenerationBatchResult, EmbeddingBatchResult],
    launch_done: Optional[threading.Event] = None,
):
    if batch.forward_mode.is_decode():
        self.process_batch_result_decode(batch, result, launch_done)
    elif batch.forward_mode.is_extend():
        self.process_batch_result_prefill(batch, result, launch_done)
    elif batch.forward_mode.is_idle():
        if self.enable_overlap:
            self.tp_worker.resolve_last_batch_result(launch_done)
            if batch.next_batch_sampling_info:
                batch.next_batch_sampling_info.update_regex_vocab_mask()
                self.current_stream.synchronize()
                batch.next_batch_sampling_info.sampling_info_done.set()
    elif batch.forward_mode.is_dummy_first():
        batch.next_batch_sampling_info.update_regex_vocab_mask()
        self.current_stream.synchronize()
        batch.next_batch_sampling_info.sampling_info_done.set()
```

## Overlap

<https://github.com/sgl-project/sglang/blob/85e1a6f3aa5a2288ca85fe3fe922c733b6533fa7/python/sglang/srt/managers/scheduler.py#L399>

initialize a `deque`

<https://github.com/sgl-project/sglang/pull/1677/>

<https://github.com/sgl-project/sglang/pull/1687/>

## TP Worker

每个 Scheduler 都有一个 Worker

`Reqs` ⇒ `ScheduleBatch` ⇒ `ModelWorkerBatch` ⇒ `ForwardBatch`

初始化

- `ModelConfig`
- `ModelRunner` : 将`ModelConfig` 传入

目的：提供一层抽象：将`ModelWorkerBatch` 转化为`ForwardBatch` 的

主要函数：

`forward_batch_generation`

1. 从`ModelWorkerBatch`获得`ForwardBatch` : 获得一次 Forward 的所有信息
   - 包含`AttentionBackend`

- 使用`ModelRunner` `forward` 这个`forward_batch`

- 调用`ModelRunner` `sample`

## ModelRunner

真正模型前向计算的地方

在`init`过程中，会启动

1. model
2. init memory

model forward 的统一接口

### Model

模型的创建与加载，SGLang 所支持的模型都位于<https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models>

其中一些模型常用到的`layer` 都是自定义写好的，位于<https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers>

这些 model 类除了正常的 init, forward 定义外，都统一定义了`load_weights` 来加载权重

最后定义`EntryClass` 来表示入口的类

用来建立一个`str ⇒ Class`的映射，为了模型架构的加载

核心的`radix attention`:

基于不同的`attn_backend`来做对应的`forward`

```python
    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

        return forward_batch.attn_backend.forward(
            q,
            k,
            v,
            self,
            forward_batch,
            save_kv_cache,
            **kwargs,
        )
```

# Load Balance

1. ROUND_ROBIN

```python
  self.round_robin_counter = 0

  def round_robin_scheduler(self, req):
      self.workers[self.round_robin_counter].send_pyobj(req)
      self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)
```

2. SHORTEST_QUEUE
