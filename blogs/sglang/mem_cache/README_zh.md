# 内存分配与缓存管理

[Orginal Version(English)](./README.md)

执行流程：
`launch_server` ⇒ `_launch_subprocesses` ⇒ `Init Scheduler` ⇒ `Init TpWorker` ⇒ `Init ModelConfig & ModelRunner` ⇒ `ModelRunner init KV Cache Pool & Allcator`

主要包含以下要点：

1. `KV Cache`初始化中`mem-fraction-static`的工作原理
2. 每个token的KV缓存如何计算
3. KV缓存池的管理机制（分配、释放、使用）
4. Radix Tree是如何管理和复用`KV Cache`

有以下两个章节
​​- `KV Cache`管理​​：探讨如何通过分配、释放和使用来管理`KV Cache`
​- ​`Radix Tree Cache`​​：探讨基数树数据结构如何实现KV缓存复用

## `KV Cache`管理
>
> ​​背景知识​​
ModelRunner：持有实际模型，负责执行模型的​​前向传播​

以下是ModelRunner的初始化过程，同时也是KV缓存池的初始化过程

在初始化内存池时，SGLang提供了三个抽象管理器：

req_to_token_pool：将请求的token映射到out_cache_loc的内存池
token_to_kv_pool：将req_token_pool中的out_cache_loc映射到实际KV缓存数据
token_to_kv_pool_allocator：分配和释放实际KV缓存数据

```python
class ModelRunner:
  def __init__(self, model_config, ....):
    # 调整`AttentionBackend`和`mem_fraction_static`
    model_specific_adjustment()

    # 由于SGLang会根据模型架构调整设置，因此需要全局更新这些信息
    global_server_args_dict.update({...})

    # 为后续通信构建WORLD_GROUP、TP_GROUP、PP_GROUP
    # 初始化分布式设置后，获取全局最小的GPU内存
    min_per_gpu_memory = init_torch_distributed()

    initialize(min_per_gpu_memory)

  def initialize(min_per_gpu_memory):
    # 加载采样器和模型
    sampler = Sampler()
    load_model()

    ######
    # 至此，模型权重和分布式初始化已占用部分GPU内存
    # 注意：但`min_per_gpu_memory`不会变化
    ######

    # 本文核心!!!
    init_memory_pool(
      min_per_gpu_memory,
      server_args.max_running_requests,  # 这两个参数由用户设置
      server_args.max_total_tokens)

    # ...
    init_cublas()
    init_attention_backend()
    init_cuda_graphs()

  def init_memory_pool(
       total_gpu_memory,
       max_num_reqs=None,
       max_total_tokens=None):
    # 计算每个GPU可以保存多少token的KV缓存
    max_total_num_tokens = profile_max_num_token(total_gpu_memory)

    # 调整max_num_requests
    if max_num_reqs is None:
      max_num_reqs = min(
       max(max_total_num_tokens / model_config.context_len * 512, 2048),
       4096
    )

    # 调整max_total_tokens
    if max_total_tokens is None:
      if max_total_tokens > max_total_num_tokens: logger.warning...
      max_total_num_tokens = min(max_total_tokens, max_total_num_tokens)

    # 按页大小对齐
    max_total_num_tokens = (max_total_num_tokens // page_size) * page_size

    # 初始化req_to_token_pool
    req_to_token_pool = ReqToTokenPool(
           max_num_reqs + 1,
           model_config.context_len + 4,
           ...)

    # 初始化token_to_kv_pool
    token_to_kv_pool = MHATokenToKVPool(
           max_total_num_tokens,
           page_size,
           kv_cache_dtype,
           head_num,
           head_dim,
           layer_num,
           ...)

    # 初始化token_to_kv_pool_allocator
    token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        max_total_num_tokens,
        kv_cache_dtype,
        device,
        token_to_kv_pool)

    ...END !!!

  def profile_max_num_token(total_gpu_memory):
    # 获取全局最小的可用GPU内存
    # 注意：此时模型已加载
    available_gpu_memory = get_available_gpu_memory(distributed=True)

    # 计算单个token的KV缓存占用的GPU内存
    # 注意：在TP设置中，每个GPU仅处理部分`attention head`计算注意力分数
    cell_size = (
      model_config.get_num_kv_heads(get_attention_tp_size())  # 获取TP设置下的num_kv_heads数量
     * model_config.head_dim
     * num_layers
     * 2  # 因为包含K和V
     * element_size(kv_cache_dtype)  # KV缓存类型每个元素的字节数
    )

    # 这是`mem_fraction_static`的核心作用
    # 注意：
    # - `total_gpu_memory`是初始化分布式环境后的min_per_gpu_memory
    # - `available_gpu_memory`是初始化分布式环境并加载模型后的min_per_gpu_memory
    # - `total_gpu_memory * (1 - mem_fraction_static)`：其他潜在的GPU内存使用（如前向传播中的`activation`）
    # - `rest_memory`：加载模型后的空闲GPU内存减去其他GPU内存，剩余部分用于`KV缓存`
    rest_memory = available_gpu_memory - total_gpu_memory *
       (1 - mem_fraction_static)

    # 将rest_memory从GB转换为字节单位
    # 计算可以保存多少token的KV缓存
    max_num_tokens = int(rest_memory * (1 << 30) // cell_size)
    return max_num_tokens
```

通过上述简化代码，我们可以看出：

**mem_fraction_static的作用**: mem_fraction_static用于划分GPU内存给模型权重和KV缓存池。如果遇到内存不足错误，可以使用更小的值。具体流程如下：

1. 获取空闲GPU内存（M1：总空闲GPU内存）
2. 加载模型（占用部分GPU内存）
3. 再次获取空闲GPU内存（M2：加载模型后的空闲内存）
4. 计算非静态GPU内存：M3 = M1 * (1 - mem_fraction_static)
5. KV缓存池的内存：M2 - M3

**单个token的KV缓存计算方式**： tp_num_head \* head_dim \* num_layers \* 2 \* element_size (torch._utils._element_size(kv_cache_dtype))

### Managers

#### req_to_token_pool

将请求映射到其token位置的内存池。

形状：max_num_reqs + 1 × self.model_config.context_len + 4

数据类型：torch.int32

访问方式：

- dim0：具体的req_idx
- dim1：请求中的token位置（从0, 1, 2...开始），标识请求中的特定token
- 值(out_cache_loc)：指向与dim0和dim1标识的token关联的KV缓存索引

```python
class ReqToTokenPool:
  def __init__(size, max_context_len):
    req_to_token = torch.zeros(size, max_context_len, dtype=torch.int32)
    # 记录空闲槽位
    free_slots = list(range(size))

  def write(indices, values):
    req_to_token[indices] = values

  def avaiable_size():
    return len(free_slots)

  def alloc(need_size):
    if need_size > len(free_slots): return None
    # 直接移除`need_size`个槽位
    select_index = free_slots[:need_size]
        free_slots = free_slots[need_size:]
        return select_index

    def free(free_index):
      free_slots.extend(free_index)

  def clear():
    free_flost = list(range(size)
```

#### token_to_kv_pool

将req_token_pool中的out_cache_loc映射到实际KV缓存数据

主要维护k_buffer和v_buffer，两者形状相同

形状（Tensor列表）：layer_num × [Tensor]，其中每个Tensor：max_total_num_tokens + page_size × head_num × head_dim

访问方式：

- dim0：layer_id标识特定层
- dim1：out_cache_loc标识特定KV缓存索引
- dim2：head
- dim3：head_dim
- 值：实际KV缓存数据

```python
class MHATokenToKVPool(KVCache):
  def __init__(size, page_size, dtype, head_num, head_dim, layer_num, device, start_layer...):
    # 创建实际KV缓存缓冲区
    _create_buffers()
    ############
    # 此时，每个GPU内存几乎耗尽
    ###########

  def _create_buffers():
    k_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim),
                    kv_cache_dtype,
                    device,
                )
                for _ in range(layer_num)
            ]
        v_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim),
                    kv_cache_dtype,
                    device,
                )
                for _ in range(layer_num)
            ]
     def _clear_buffers():
       del k_buffer, v_buffer

   ################
   ## 读取API
   ################
   def get_key_buffer(layer_id):
     return k_buffer[layer_id - start_layer]

   def get_value_buffer(layer_id):
     return v_buffer[layer_id - start_layer]

   def get_kv_buffer(layer_id):
        return get_key_buffer(layer_id), get_value_buffer(layer_id)

    ############
    ## 写入API
    ############
    def set_kv_buffer(layer, loc, cache_k, cache_v, ...):
      layer_id = layer.layer_id
      k_buffer[layer_id - start_layer][loc] = cache_k
         v_buffer[layer_id - start_layer][loc] = cache_v
```

#### token_to_kv_pool_allocator

用于分配实际KV缓存数据：out_cache_loc

```python
class TokenToKVPoolAllocator:
  def __init__(size [max_total_num_tokens], dtype, page_size device, kvcache [token_to_kvcache_pool]):
    page_size = 1
    clear()

  def clear():
    free_slots = torch.arange(1, self.size + 1, dtype=torch.int64, device)

  def available_size():
    return len(free_slots)

  ##########################
  # 分配API
   #########################
  def alloc(need_size):
    if need_size > len(self.free_slots): return None
        select_index = free_slots[:need_size]
        free_slots = free_slots[need_size:]
        return select_index

    ###########################
    ## 释放API
    ###########################
    def free(free_index):
     free_slots = torch.cat((free_slots, free_index))
```

**为请求和out_cache_loc分配槽位**
这就引出了一个问题：SGLang如何使用上述管理器高效地为每个请求中的token分配槽位并及时释放？

LLM推理包含两个主要阶段。我们首先确定每个阶段的分配需求。

1. ​​预填充（prefill）​​：
    1. req_to_token_pool.alloc：因为有新请求
    2. token_to_kv_pool_allocator.alloc：可能，
        1. 如果请求中的token已有KV缓存，可以直接使用req_to_token_pool.write复用这些KV缓存
        2. 如果没有KV缓存，则调用token_to_kv_pool_allocator.alloc获取out_cache_loc，然后将其写入req_token_pool
1. ​​解码（decode）​​：
    1. req_to_token_pool.alloc：不需要
    2. token_to_kv_pool_allocate.alloc：需要，因为每次解码一个新token

因此，在scheduler.get_next_batch_to_run中获取ScheduleBatch时，不同阶段有不同的逻辑来处理分配和释放槽位。

```python
class ScheduleBatch:
    """存储调度器上一批次的所有信息"""

  def prepare_for_extend():
    bs = len(reqs)
    req_pool_indices = alloc_req_slots(bs)

    # fill_ids = origin_input_ids + output_ids
    # input_ids是需要计算KV缓存的token_ids
    input_ids = [r.fill_ids[len(r.prefix_indices): ] for r in reqs]

    # 这是需要分配槽位以容纳的token数量
    extend_num_tokens = sum(len(ids) for ids in input_ids)

    seq_lens = [len(r.fill_ids) for r in reqs]
    prefix_lens = [len(r.prefix_indices) for r in reqs]

    # extend_lens实际上等于`seq_lens - prefix_lens`
    extend_lens = [r.extend_input_len for r in reqs]

    for i, (req, seq_len, pre_len) in enumerate(reqs, seq_lens, pre_lens):
      req.req_pool_idx = req_pool_indices[i]

      # 再次确认
      assert seq_len - pre_len == req.extend_input_len

      if pre_len > 0:
        # 将缓存的`out_cache_loc`写入`req_to_token_pool`
        req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, pre_len)), req.prefix_indices
                )

       out_cache_loc = alloc_token_slots(extend_num_tokens)

       pt = 0
       for i in range(bs):
         # 将未缓存的`out_cache_loc`写入`req_to_token_pool`
            for i in range(bs):
                self.req_to_token_pool.write(
                    (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
                    out_cache_loc[pt : pt + extend_lens[i]],
                )
                pt += extend_lens[i]
       ... END !!!

  def prepare_for_decode():
    bs = len(reqs)

    # 分配`bs`个token
    out_cache_loc = self.alloc_token_slots(bs)

    # 计算`req_to_token_pool`位置
    locs = seq_lens + 1

    # 写入
    req_to_token_pool.write(
            (req_pool_indices, locs), out_cache_loc.to(torch.int32)
        )
       ... END !!!

  def alloc_req_slots(num_reqs):
    req_pool_indices = req_to_token_pool.alloc(num_reqs)
    if req_pool_indices is None: raise RuntimeError("")
    return req_pool_indices

  def alloc_token_slots(num_tokens):
    out_cache_loc = self.token_to_kv_pool_allocator.alloc(num_tokens)
    if out_cache_loc is None: raise RuntimeError()
    return out_cache_loc
```

**计算注意力分数时读取和保存实际KV缓存数据**
在前向传播中，model_runner会调用attention_backnend.init_forward_metadata初始化注意力后端的元数据，然后调用实际的forward_extend和forward_decode

在init_forward_metadata中，通过req_to_token_pool.req_to_token获取页表，用于每层注意力分数的计算

```python
class FlashAttentionBackend(AttentionBackend):
  def init_forward_metadata(forward_batch):
    metadata = FlashAttentionMetadata()
    if forward_batch.is_decode():
      metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
      # 获取页表！
      metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                 forward_batch.req_pool_indices, : metadata.max_seq_len_k
             ]
     elif forward_batch.is_extend():
       # ... 几乎相同 ...
```

保存和检索过程发生在模型前向传播中，即attention_backend.forward_extend或attention_backend.forward_extend

```python
class FlashAttention(AttentionBackend):
  def forward_extend(q, k, v, layer, forward_batch, save_kv_cache=True, ...):
    if k is not None:
      if v is not None:
        cache_loc = forward_batch.out_cache_loc

        # !!! 将KV缓存保存到token_to_kv_pool !!!
        forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, ...
                    )
       # 使用所有层预计算的元数据
        # 为FlashAttention操作准备元数据
        metadata = self.forward_metadata
        page_table = metadata.page_table
        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens = metadata.cache_seqlens_int32
        max_seqlen_q = metadata.max_seq_len_q
        max_seqlen_k = metadata.max_seq_len_k
        cu_seqlens_k = metadata.cu_seqlens_k

        # !!! 从token_to_kv_pool检索KV缓存 !!!
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
        # 检查格式
        key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
        value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

        result = flash_attn_with_kvcache(
          q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
          key_cache,
          value_cache,
          page_table,
          ...
       )

       return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

  def forward_decode(forward_batch):
    # ... 几乎与forward_extend相同 ...
```

第一部分KV缓存管理到此结束，我们讨论了：

KV缓存如何初始化
KV缓存如何管理（为请求分配槽位和token）
计算注意力分数时如何保存和检索实际KV缓存数据

## Radix Tree Cache

SGLang的一个创新思想是基数注意力，它使用基数树尽可能复用KV缓存

那么，什么是基数树？

其核心思想是获取前缀

### Radix Tree

```python
class TreeNode:
    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)  # 使用1页大小的key作为字典键
        self.parent = None
        self.key = None  # Key是`token_ids`
        self.value = None  # Value是`out_cache_loc`，记录实际KV缓存数据的位置

        self.lock_ref = 0  # 有多少请求引用此节点

        self.last_access_time = time.monotonic()

        self.hit_count = 0

        # 表示节点正在从主机加载KV缓存
        self.loading = False

        # 存储KV缓存的主机索引
        self.host_value = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

class RadixTree(BasePrefixCache):
  def __init__(req_to_token_pool, token_to_kv_pool_allocator, page_size, ...):
    if page_size == 1:
      # key_match_fn：给定两个key，返回它们共有的前缀ids数量
            key_match_fn = _key_match_page_size1

            # get_child_key_fn：获取1页大小的key
            get_child_key_fn = lambda key: key[0]
        else:
            key_match_fn = partial(_key_match_paged, page_size=page_size)
            get_child_key_fn = lambda key: tuple(key[:page_size])
    reset()

  def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._record_all_cleared_event()
```

#### 匹配

```python
  ########################
   # 匹配前缀
   ########################
   def match_prefix(key: List[int]):
     page_aligned_len = len(key) // page_size * page_size
       key = key[:page_aligned_len]

       value, last_node = _match_prefix_helper(root_node, key)
       if value: value = torch.cat(value)
       else: value = torch.empty((0,), dtype=torch.int64, device=device)

       # 1. 基数树中的前缀`out_cache_loc`
       # 2. last_node
      return value, last_node

  def _match_prefix_helper(node, key):
    # 更新时间
    node.last_access_time = time.monotonic()

    # 先获取子key
    child_key = self.get_child_key_fn(key)

    value = []
    while len(key) > 0 and child_key in node.children.keys():

      child = node.children[child_key]

      # 更新时间
      child.last_access_time = time.monotonic()

      # 获取前缀ids的数量（n * page_size）
      prefix_len = self.key_match_fn(child.key, key)

      if prefix_len < len(child.key):
        # 不完全匹配，拆分一个完全匹配但更短的new_node

        # 注意：prefix_len至少为1页大小，因为`child_key in node.children.keys()`
        new_node = self._split_node(child.key, child, prefix_len)

        # 追加匹配的值
        value.append(new_node.value)
               node = new_node
               break
      else:
        # 完全匹配，尝试获取下一个子节点

        # 保存值
        value.append(child.value)

        # 更新节点
               node = child

               # 截断已匹配的前缀key
               key = key[prefix_len:]

               if len(key):
                 child_key = self.get_child_key_fn(key)
       return value, node
```

拆分节点：

```
  #############
   # 拆分节点
   #############
  def _split_node(key: List[int], child, split_len):
    # 这里的key实际上是子节点的key
    # key和value将被分成两部分
    # key和value: [......................... | ..........................]
    #                                       prefix_len
    #                  左侧：新节点的kv        右侧：截断的子节点
    # 拆分后，`child(node)`将变为
    # `parent <-> child`    =>
    # `parent <-> new_node <-> truncated child`

    # 创建新节点
    new_node = TreeNode()

    # 使`new_node ---截断子节点的1页大小key---> child`
    new_node.children = {self.get_child_key_fn(key[split_len:]): child}

       # 使`parent -> new_node`
       new_node.parent = child.parent

       # 使new_node获得相同的引用计数
       new_node.lock_ref = child.lock_ref

       # 获取左侧kv，并设置给new_node
       new_node.key = child.key[:split_len]
       new_node.value = child.value[:split_len]

    # 使`new_node <- child`
       child.parent = new_node

       # 使`child`变为`截断的子节点`：截断split_len的key和value
       child.key = child.key[split_len:]
       child.value = child.value[split_len:]

       # 使`parent ----new_node的1页大小key---> new_node
       new_node.parent.children[self.get_child_key_fn(key)] = new_node

    return new_node
```

#### 插入节点

```python
 ################
 # 插入节点
 ################
 def insert(self, key: List, value=None):
     if self.disable: return 0

     if value is None: value = [x for x in key]

     return _insert_helper(root_node, key, value)

  def _insert_helper(node, key, value):
    # 更新节点时间用于LRU淘汰
    node.last_access_time = time.monotonic()

      if len(key) == 0: return 0

      # 获取用于搜索前缀的1页大小key
      child_key = get_child_key_fn(key)

      total_prefix_length = 0

      while len(key) > 0 and child_key in node.children.keys():
      # 获取下一个节点
      node = node.children[child_key]
      # 更新下一个节点的时间
      node.last_access_time = time.monotonic()

      # 获取下一个节点和查询key的前缀长度
      prefix_len = self.key_match_fn(node.key, key)

      total_prefix_length += prefix_len

      # 更新key和value
      key = key[prefix_len:]
          value = value[prefix_len:]

          if prefix_len < len(node.key):
            # 不完全匹配，拆分节点
            new_node = _split_node(node.key, node, prefix_len)

              node = new_node

          if len(key):
            # 仍有部分key未匹配，尝试继续查找下一个节点
            child_key = get_child_key_fn(key)

            # 注意：如果prefix_len < len(node.key)
            # 则无法继续此while循环
            # 因为拆分后的新节点只有一个子节点，即未匹配的节点
            # 所以这个新的`child_key`不在`node.children.keys()`中
            # 此while循环仅在完全匹配但查询key仍有剩余部分时继续

   if len(key):
     # 如果仍有未匹配的剩余key，
     # 创建新节点
     # 注意：此新节点的lock_ref为0，因此可被淘汰
     new_node = TreeNode()
          new_node.parent = node
          new_node.key = key
          new_node.value = value

          # 使node`指向此`new_node`
          node.children[child_key] = new_node

          # 这是可淘汰的，因为它是叶节点
          evictable_size_ += len(value)

   return total_prefix_length
```

#### API

- 请求完成或未完成时的缓存
- 删除不需要的缓存

```python
 #######################
 # 缓存未完成的请求
  #######################
  def cache_unfinished_req(req):
    token_ids = req.fill_ids

    # 获取`out_cache_loc`，即Value
    kv_indices = req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
      ]

      if page_size != 1:
        page_aligned_len = len(kv_indices) // page_size * page_size
        # 对齐V
          page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
      else:
          page_aligned_len = len(kv_indices)
          page_aligned_kv_indices = kv_indices.clone()

      # 对齐K
      page_aligned_token_ids = token_ids[:page_aligned_len]

      # 插入K,V
      new_prefix_len = insert(page_aligned_token_ids, page_aligned_kv_indices)

      # 移除重复部分
      token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
      )

      # 获取前缀`out_cache_loc`和`new_last_node`
      new_indices, new_last_node = self.match_prefix(page_aligned_token_ids)

      # 仅写入新的`out_cache_loc`
      req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
      )

      # root -> ... -> last_node -> ... -> new_last_node
      # |-- lock_ref - 1 --|
      dec_lock_ref(req.last_node)

      # root -> ... -> last_node -> ... -> new_last_node
      # |------------- lock_ref + 1 -----------------|
      inc_lock_ref(new_last_node)


 #####################
 # 缓存完成的请求
 #####################
  def cache_finished_req(req):
   if self.disable:
     # 如果禁用基数树，直接释放此完成请求的KV缓存

     # 获取`out_cache_loc`
     kv_indices = req_to_token_pool.req_to_token[
              req.req_pool_idx, : len(req.origin_input_ids) + len(req.output_ids) - 1
          ]

          # 释放`req槽位`和`token_to_kv_pool槽位`
          token_to_kv_pool_allocator.free(kv_indices)
          req_to_token_pool.free(req.req_pool_idx)
          return

     # 如果使用基数树，不立即释放KV缓存以便复用

     # 获取token_ids，即key
     token_ids = (req.origin_input_ids + req.output_ids)[:-1]

     # 获取`out_cache_loc`，即value
     kv_indices = req_to_token_pool.req_to_token[
        req.req_pool_idx, : len(token_ids)
    ]

    # 假设页大小为1，因此自动对齐
    page_aligned_len = len(kv_indices)
     page_aligned_kv_indices = kv_indices.clone()

    # 将[token_ids, out_cache_loc]插入基数树以便复用
    new_prefix_len = insert(
         token_ids[:page_aligned_len], page_aligned_kv_indices
    )

     # 仅释放[len(prefix_indices): new_prefix_len]部分的kv池，为什么？
     # 因为这部分`out_cache_loc`是重复的（冗余的）！

     # 整个过程如下：
     # `req.prefix_indices`在首次调度时计算
     # `new_prefix_len`是完成时的前缀长度
     # [len(req.prefix_indices): new_prefix_len]是计算过程中重复的部分
    token_to_kv_pool_allocator.free(
          kv_indices[len(req.prefix_indices) : new_prefix_len]
     )

     # 释放`req槽位`
     # 因为请求已完成，其req_pool_idx可用于其他请求
     req_to_token_pool.free(req.req_pool_idx)

     # 减少拥有out_cache_loc[:len(prefix_indices)]的节点的lock_ref
     # 这些部分可能变为可淘汰
     # 但注意：这些`out_cache_loc`尚未被淘汰
     dec_lock_ref(req.last_node)
```

```python
  def evict(num_tokens: int):
    if disable: return

    leaves = _collect_leaves()

    # 按`last_access_time`排序（LRU）
    heapq.heapify(leaves)

    num_evicted = 0
    while num_evicted < num_tokens and len(leaves):
      x = heapq.heappop(leaves)
      if x == self.root_node: break

      # 如果有请求指向此节点，跳过
            if x.lock_ref > 0: continue

            # 释放此节点的`out_cache_loc`
            token_to_kv_pool_allocator.free(x.value)

            num_evicted += len(x.value)
            _delete_leaf(x)

            # 为下一次淘汰添加新的叶节点
            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

  def _delete_leaf(node):

    # 从父节点中删除此节点
    for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]

        # 更新可淘汰大小
        evictable_size_ -= len(node.key)

```

-- --
**使用方式**

1. 当prefill结束时，

```python
def process_batch_result_prefill(batch, result):
  for i, (req, next_token_id) in enumerate(batch.reqs, result.next_token_ids):
    req.output_ids.append(next_token_id)
        req.check_finished()

        if req.finished():
          tree_cache.cache_finished_req(req)

       elif not batch.decoding_reqs or req not in batch.decoding_reqs:
            # 更新基数树以便其他请求匹配
            tree_cache.cache_unfinished_req(req)
```

2. 当decode结束时，

```python
def process_batch_result_decode(batch, result):
  for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
    req.check_finished()

    if req.finished():
           tree_cache.cache_finished_req(req)
```

<aside> 💡
只有在decode完成时，tree_cache才会缓存其（token_ids, out_cache_loc）

</aside>

**删除不需要的缓存**:
当token_to_kv_pool中的available_size无法支持传入请求时，会发生淘汰（即释放out_cache_loc）

```python
def alloc_token_slots(num_tokens: int, backup_state: bool = False):
    if token_to_kv_pool_allocator.available_size() < num_tokens:
      if tree_cache is not None:
          tree_cache.evict(num_tokens)

  out_cache_loc = token_to_kv_pool_allocator.alloc(num_tokens)
```

## 参考

- [https://hebiao064.github.io/fa3-attn-backend-basic](https://hebiao064.github.io/fa3-attn-backend-basic)
