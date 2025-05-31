# Memory Cache in SGLang
>
> Note: Complex systems often include numerous corner cases and technical implementations that can make the source code challenging to understand for newcomers.
>
> To make the core concepts more accessible, this blog post uses pseudocode that focuses on the main ideas while omitting implementation details (such as self references and other technical specifics). While simplified, the pseudocode maintains the essential logic and workflow of the system.
>
> Of source, if you want to know all details, the best way is to look directly at the source code, which is available in [here](https://github.com/sgl-project/sglang)

Main walker:

`launch_server` ⇒ `_launch_subprocesses` ⇒ `Init Scheduler` ⇒ `Init TpWorker` ⇒ `Init ModelConfig & ModelRunner` ⇒ `ModelRunner init KV Cache Pool & Allcator`

Main points in this blog:

- How `mem-fraction-static` works in the KV Cache Initiation
- How is each token’s `KV Cache` computed
- How `KV Cache Pool` are managed(allocate, free, use)
- How `Radix Cache` reuses KV Cache

This blog mainly compasses 2 sections

- In the KV Cache Management section, we will explore how `KV Cache` is managed through allocation, freeing, and usage
- In the Radix Tree Cache section, we will explore how the `radix tree` data structure enables KV Cache reuse

# KV Cache Management

> **Background**

The `ModelRunner`: owns the real model, runs the **forward** pass of the models
>

here is the initialization of `ModelRunner` , and also the initialization of `KV Cache Pool`

In this process of initating `memory pool` , SGLang provides 3 abstract managers

1. `req_to_token_pool`: A memory pool that maps a request’s tokens to `out_cache_loc`
2. `token_to_kv_pool`: A pool that maps `out_cache_loc` from `req_token_pool` to its real KV Cache data
3. `token_to_kv_pool_allocator`: Allocate and free real KV Cache data

```python
class ModelRunner:
  def __init__(self, model_config, ....):
    
    # adjust `AttentionBackend`, `mem_fraction_static`
    model_specific_adjustment()
    
    # since SGLang adjusts the settings depending on Model Arch 
    # then update that info globally
    global_server_args_dict.update({...})
    
    # build WORLD_GROUP, TP_GROUP, PP_GROUP for later communication
    # after init the distibuted settings, get the minimum GPU memory across the world
    min_per_gpu_memory = init_torch_distributed()
    
    initialize(min_per_gpu_memory)
  
  def initialize(min_per_gpu_memory):
    
    # load sampler and model
    sampler = Sampler()
    load_model()
    
    ######
    # Until now, Model Weights & Distributed Initialization occpuy some GPU memory
    # Note: but `min_per_gpu_memory` doesn't change
    ######
    
    # Core in this blog!!!
    init_memory_pool(
      min_per_gpu_memory, 
      server_args.max_running_requests, # these 2 args are set by users
      server_args.max_total_tokens)
    
    # ...
    init_cublas()
    init_attention_backend()
    init_cuda_graphs()
  
  def init_memory_pool(
       total_gpu_memory, 
       max_num_reqs=None,
       max_total_tokens=None):
    # compute how many token's KV Cache can be saved in each GPU
    max_total_num_tokens = profile_max_num_token(total_gpu_memory)
    
    # adjust max_num_requests
    if max_num_reqs is None:
      max_num_reqs = min(
       max(max_total_num_tokens / model_config.context_len * 512, 2048), 
       4096
    )
    
    # adjust max_total_tokens
    if max_total_tokens is None:
      if max_total_tokens > max_total_num_tokens: logger.warning...
      max_total_num_tokens = min(max_total_tokens, max_total_num_tokens)
    
    # align page size
    max_total_num_tokens = (max_total_num_tokens // page_size) * page_size
    
    # init req_to_token_pool
    req_to_token_pool = ReqToTokenPool(
           max_num_reqs + 1,
           model_config.context_len + 4,
           ...)
           
    # init token_to_kv_pool
    token_to_kv_pool = MHATokenToKVPool(
           max_total_num_tokens,
           page_size,
           kv_cache_dtype,
           head_num,
           head_dim,
           layer_num,
           ...)
     
    # init token_to_kv_pool_allocator
    token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        max_total_num_tokens,
        kv_cache_dtype,
        device,
        token_to_kv_pool)
      
    ...END !!!  
  
  def profile_max_num_token(total_gpu_memory):
    # get min_per_gpu_memory in the world
    # Note: model has been loaded before
    available_gpu_memory = get_available_gpu_memory(distributed=True)
    
    # Compute how much gpu memory **a token's KV Cache** occupy
    # Note: In TP settings, each GPU only handles part of `attention head` when computing attention scores
    cell_size = (
      model_config.get_num_kv_heads(get_attention_tp_size()) # get how many num_kv_heads in TP setting
     * model_config.head_dim
     * num_layers
     * 2 # since K and V
     * element_size(kv_cache_dtype) # bytes for each element of KV Cache Type
    )
    
    # This is the **role** of `mem_fraction_static` here
    # Note: 
    # - `total_gpu_memory` is after initializing the distributed environment, min_per_gpu_memory
    # - `available_gpu_memory` is after initializing the distbuted environment and loading model, min_per_gpu_memory 
    # - `total_gpu_memory * (1 - mem_fraction_static)`: the other potential GPU memory usage (like `activation` in the forward pass)
    # - `rest_memory`: Free GPU Memory(after loading model) substracting the other GPU memory, the rest is for `KV Cache`
    rest_memory = available_gpu_memory - total_gpu_memory * 
       (1 - mem_fraction_static)
       
    # convert rest_memory from GigeByte back to Byte metric
    # compute how many tokens' KV cache can be saved
    max_num_tokens = int(rest_memory * (1 << 30) // cell_size)
    return max_num_tokens
```

Reading from above simplified code reviews, we can see:

1. `mem_fraction_static` ’s usage

The `mem_fraction_static` of `GPU memory` is used for `model weights` and `KV Cache Pool`, Use a smaller value if you see out-of-memory errors. But how does the process go?

1. Get Free GPU Memory  (`M1`: total GPU free memory)
2. Load model (this occupy some GPU Memory)
3. Get Free GPU Memory again (`M2`: After Loading Model)
4. Compute non-static GPU memory: (`M3 = M1 * (1 - mem_fraction_static)` )
5. The memory for KV cache Pool: `M2 - M3`

1. How a token’s KV Cache is computed:

`tp_num_head * head_dim * num_layers * 2 * element_size (torch._utils._element_size(kv_cache_dtype))`

## Managers

### req_to_token_pool

A memory pool that maps a request to its token locations.

Shape: `max_num_reqs *+* 1`  * `self.model_config.context_len *+* 4`

Dtype: `torch.int32`

Access:

- dim0: the concrete `req_idx`
- dim1: token positions in req (starting from 0, 1, 2…), identify the specific token in the request
- `out_cache_loc` for token, it points to the KV cache indices associated with the token identified by dim0 and dim1

```python
class ReqToTokenPool:
  def __init__(size, max_context_len):
    req_to_token = torch.zeros(size, max_context_len, dtype=torch.int32)
    # record free slots
    free_slots = list(range(size))
  
  def write(indices, values):
    req_to_token[indices] = values
    
  def avaiable_size():
    return len(free_slots)
    
  def alloc(need_size):
    if need_size > len(free_slots): return None
    # directly remove `need_size` slots
    select_index = free_slots[:need_size]
        free_slots = free_slots[need_size:]
        return select_index
    
    def free(free_index): 
      free_slots.extend(free_index)
  
  def clear(): 
    free_flost = list(range(size)
```

### token_to_kv_pool

A pool that maps `out_cache_loc` from `req_token_pool` to its real KV Cache data

Mainly maintain the `k_buffer` and `v_buffer` which has the same shape

Shape(List of `Tensor`): `layer_num` *[ `Tensor` ], where each `Tensor`: `max_total_num_tokens + page_size`* `head_num`  * `head_dim`

Access:

- dim0: `layer_id` identify the specific layer
- dim1: `out_cache_loc` identify the specific KV cache indices
- dim2: `head`
- dim3: `head_dim`
- value: real KV Cache data

```python
class MHATokenToKVPool(KVCache):
  def __init__(size, page_size, dtype, head_num, head_dim, layer_num, device, start_layer...):
    # create real KV Cache buffers
    _create_buffers()
    ############
    # Now, each GPU Memory is nearly exhausted
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
   ## READ API
   ################
   def get_key_buffer(layer_id):
     return k_buffer[layer_id - start_layer]
  
   def get_value_buffer(layer_id):
     return v_buffer[layer_id - start_layer]
     
   def get_kv_buffer(layer_id):
        return get_key_buffer(layer_id), get_value_buffer(layer_id)
    
    ############
    ## WRITE API
    ############
    def set_kv_buffer(layer, loc, cache_k, cache_v, ...):
      layer_id = layer.layer_id
      k_buffer[layer_id - start_layer][loc] = cache_k
         v_buffer[layer_id - start_layer][loc] = cache_v
```

### token_to_kv_pool_allocator

用于在Scheduler_info中分配

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
  # ALLOCATE API
   #########################
  def alloc(need_size):
    if need_size > len(self.free_slots): return None
        select_index = free_slots[:need_size]
        free_slots = free_slots[need_size:]
        return select_index
    
    ###########################
    ## FREE API
    ###########################
    def free(free_index):
     free_slots = torch.cat((free_slots, free_index))
```

## Allocate Slots to Reqs & Out_cache_loc

Now comes the question, how `SGLang` use above managers to efficiently `allocate` slots for each token in reqs and `free` some in time.

LLMs Inference mainly comprise 2 stages, we can start from that and think in each stage what should we allocate

1. prefill:
    1. `req_to_token_pool.alloc` : since we have new reqs
    2. `token_to_kv_pool_allocator.alloc` : Maybe,
        1. if we have the `kv cache` in the tokens in the reqs, we can just use `req_to_token_pool.write` to reuse those kv cache
        2. if we don’t have the `kv cache`, then get `out_cache_loc` by calling `token_to_kv_pool_allocator.alloc` , then write `out_cache_loc` into `req_token_pool`
2. decode:
    1. `req_to_token_pool.alloc` : don’t need
    2. `token_to_kv_pool_allocate.alloc` Need, since we decode one new token one time

So in the `scheduler.get_next_batch_to_run` where get `ScheduleBatch` , for different stage, there are different logics to prepare where allocate and free slots happened.

```python
class ScheduleBatch:
    """Store all information of a batch on the scheduler."""
  
  def prepare_for_extend():
    bs = len(reqs)
    req_pool_indices = alloc_req_slots(bs)
    
    # fill_ids = origin_input_ids + output_ids
    # input_ids are those token_ids whose KV Cache needs computing
    input_ids = [r.fill_ids[len(r.prefix_indices): ] for r in reqs]
    
    # this is the num tokens we need allocate slots to accommodate
    extend_num_tokens = sum(len(ids) for ids in input_ids)
    
    seq_lens = [len(r.fill_ids) for r in reqs]
    prefix_lens = [len(r.prefix_indices) for r in reqs]
    
    # extend_lens is actually equal to `seq_lens - prefix_lens`
    extend_lens = [r.extend_input_len for r in reqs]
    
    for i, (req, seq_len, pre_len) in enumerate(reqs, seq_lens, pre_lens):
      req.req_pool_idx = req_pool_indices[i]
      
      # here assert again
      assert seq_len - pre_len == req.extend_input_len
      
      if pre_len > 0:
        # write cached `out_cache_loc` into `req_to_token_pool`
        req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, pre_len)), req.prefix_indices
                )
        
       out_cache_loc = alloc_token_slots(extend_num_tokens)
       
       pt = 0
       for i in range(bs):
         # write uncached `out_cache_loc` into `req_to_token_pool`
            for i in range(bs):
                self.req_to_token_pool.write(
                    (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
                    out_cache_loc[pt : pt + extend_lens[i]],
                )
                pt += extend_lens[i]
       ... END !!!
  
  def prepare_for_decode():
    bs = len(reqs)
    
    # allocate `bs` tokens
    out_cache_loc = self.alloc_token_slots(bs)
    
    # compute `req_to_token_pool` locs
    locs = seq_lens + 1
    
    # write 
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

## Read & Save Real KV Cache Data when computing Attention Scores

In model forward, `model_runner` will call `attention_backnend.init_forward_metadata` to initialize the metadata for the attention backend and then call the actual `forward_extend` and `forward_decode`

during the `init_forward_metadata` , by use `req_to_token_pool.req_to_token` , we get the `page table` which is then used in each layer’s attention score computation

```python
class FlashAttentionBackend(AttentionBackend):
  def init_forward_metadata(forward_batch):
    metadata = FlashAttentionMetadata()
    if forward_batch.is_decode():
      metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
      # get the page table!
      metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                 forward_batch.req_pool_indices, : metadata.max_seq_len_k
             ]
     elif forward_batch.is_extend():
       # ... nearly same ...
```

`save & retrieve` process takes place at the model forward, where `attention_backend.forward_extend` or `attention_backend.forward_extend`

```python
class FlashAttention(AttentionBackend):
  def forward_extend(q, k, v, layer, forward_batch, save_kv_cache=True, ...):
    if k is not None:
      if v is not None:
        cache_loc = forward_batch.out_cache_loc
        
        # !!! Save the KV Cache into token_to_kv_pool !!!
        forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, ...
                    )
       # Use precomputed metadata across all layers
        # prepare metedata for FlashAttention operator
        metadata = self.forward_metadata
        page_table = metadata.page_table
        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens = metadata.cache_seqlens_int32
        max_seqlen_q = metadata.max_seq_len_q
        max_seqlen_k = metadata.max_seq_len_k
        cu_seqlens_k = metadata.cu_seqlens_k
        
        # !!! Retrive the KV Cache from token_to_kv_pool !!!
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
        # review the format
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
    # ... nearly same to forward_extend ...
```

The first section `KV Cache Management` is over here, we talked about

1. How `KV Cache` are initiated
2. How `KV Cache` is manged (allocate `slots, tokens` to reqs)
3. How the real `KV Cache data` are saved and retrieved when computing attention scores

# Radix Tree Cache

One novel idea of `SGLang` is `Radix Attention` , which uses `radix tree` to reuse `KV Cache` as much as possible.

So, what is `Radix Tree`?

Its core idea is to get prefix

## Radix Tree

```python
class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode) # use 1page-size key as the dict_key
        self.parent = None
        self.key = None # Key is the `token_ids`
        self.value = None # Value is the `out_cache_loc`, which records the location of real KV Cache data
        
        self.lock_ref = 0 # how many reqs reference this node
        
        self.last_access_time = time.monotonic()

        self.hit_count = 0
        
        # indicating the node is loading KV cache from host
        self.loading = False
        
        # store the host indices of KV cache
        self.host_value = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
```

```python
class RadixTree(BasePrefixCache):
  def __init__(req_to_token_pool, token_to_kv_pool_allocator, page_size, ...):
    if page_size == 1:
      # key_match_fn: given 2 keys, return how many prefix ids that two keys has 
            key_match_fn = _key_match_page_size1 
            
            # get_child_key_fn: get 1-page-size key
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

### Match

```python
  ########################
   # Match Prefix
   ########################
   def match_prefix(key: List[int]):
     page_aligned_len = len(key) // page_size * page_size
       key = key[:page_aligned_len]
       
       value, last_node = _match_prefix_helper(root_node, key)
       if value: value = torch.cat(value)
       else: value = torch.empty((0,), dtype=torch.int64, device=device)
       
       # 1. prefix `out_cache_loc` in the radix tree
       # 2. last_node
      return value, last_node

  def _match_prefix_helper(node, key):
    # update time
    node.last_access_time = time.monotonic()
    
    # get child key first
    child_key = self.get_child_key_fn(key)
    
    value = []
    while len(key) > 0 and child_key in node.children.keys():
    
      child = node.children[child_key]
      
      # update time
      child.last_access_time = time.monotonic()
      
      # get how many number of prefix ids (n * page_size)
      prefix_len = self.key_match_fn(child.key, key)
      
      if prefix_len < len(child.key):
        # not a full match, split a full match, but shorter new_node
        
        # NOTE: prefix_len is at least 1-page-size since `child_key in node.children.keys()`
        new_node = self._split_node(child.key, child, prefix_len)
        
        # append the matched value
        value.append(new_node.value)
               node = new_node
               break
      else:
        # full match, try to get next child
        
        # save the value
        value.append(child.value)
        
        # update the node
               node = child
               
               # truncate the prefix matched keys
               key = key[prefix_len:]
               
               if len(key):
                 child_key = self.get_child_key_fn(key)
       return value, node
```

### Split Node

```python
  #############
   # Split Node
   #############
  def _split_node(key: List[int], child, split_len):
    # here, key is actually child's key
    # key and value will be split into two parts
    # key and value: [......................... | ..........................]
    #                                       prefix_len      
    #                  left: a new node's kv        right: truncated child
    # after this split process, `child(node)` will be
    # `parent <-> child`    =>
    # `parent <-> new_node <-> truncated child`
    
    # create a new node
    new_node = TreeNode()
    
    # make `new_node ---truncated child's 1-page-size key---> child`
    new_node.children = {self.get_child_key_fn(key[split_len:]): child}
       
       # make `parent -> new_node`
       new_node.parent = child.parent
       
       # make new_node get the same ref count
       new_node.lock_ref = child.lock_ref
       
       # get left side kv, and set them to new_node
       new_node.key = child.key[:split_len]
       new_node.value = child.value[:split_len]
    
    # make `new_node <- child`
       child.parent = new_node
       
       # make `child` become `truncated child`: truncate the split_len key and value
       child.key = child.key[split_len:]
       child.value = child.value[split_len:]
       
       # make `parent ----new_node's 1-page-size key---> new_node 
       new_node.parent.children[self.get_child_key_fn(key)] = new_node
 
    return new_node
```

### Insert Node

```python
 ################
 # Insert Node
 ################
 def insert(self, key: List, value=None):
     if self.disable: return 0
     
     if value is None: value = [x for x in key]
     
     return _insert_helper(root_node, key, value)

  def _insert_helper(node, key, value):
    # update node's time for LRU eviction
    node.last_access_time = time.monotonic()
    
      if len(key) == 0: return 0
      
      # get 1-page-size key used for searching prefix
      child_key = get_child_key_fn(key)
      
      total_prefix_length = 0
      
      while len(key) > 0 and child_key in node.children.keys():
      # get next node
      node = node.children[child_key]
      # update next node's time
      node.last_access_time = time.monotonic()
      
      # get prefix_len of next node and query key
      prefix_len = self.key_match_fn(node.key, key)
      
      total_prefix_length += prefix_len
      
      # update key and value
      key = key[prefix_len:]
          value = value[prefix_len:]
          
          if prefix_len < len(node.key):
            # not a full match, split the node
            new_node = _split_node(node.key, node, prefix_len)
             
              node = new_node
          
          if len(key):
            # there are still some keys hasn't been matched, try to continue to find next node
            child_key = get_child_key_fn(key)
            
            # NOTE: if prefix_len < len(node.key)
            # then it is impossible to continue this while loop
            # because the splitted new node only have one child, which is the unmatched node
            # so this new `child_key` doesn't exist `node.children.keys()`
            # this while loop continues only if a full match, but the query key still has a remaining part
   
   if len(key):
     # if there exists still a remaining key that doesn't match in this radix tree,
     # create a new node
     # NOTE: this new node's lock_ref is 0, so it deems evictable
     new_node = TreeNode()
          new_node.parent = node
          new_node.key = key
          new_node.value = value
          
          # make node` point to this `new_node`
          node.children[child_key] = new_node
          
          # this is evictable since it is a leaf node
          evictable_size_ += len(value)
          
   return total_prefix_length  
```

### Lock Ref

```python

 ##################
 # Handle Lock Ref
 ##################
  def dec_lock_ref(node):
   if disable: return 0 # if disable radix tree
   delta = 0
   
   # bottom to up
   while node != root_node:
     if node.lock_ref == 1:
       # if there is only 1 ref to this node, this node deems evictable
           evictable_size_ += len(node.value)
             protected_size_ -= len(node.value)
             delta += len(node.value)
         lock_ref -= 1
         node = node.parent
    return delta
    
 def inc_lock_ref(node):
   if disable: return 0
   delta = 0
   
   # bottom to up
   while node != root_node:
     if node.lock_ref == 0:
       # if no other req ref this node, this node turns evictable to protectable
       evictable_size_ -= len(node.value)
             self.protected_size_ += len(node.value)
             delta -= len(node.value)
     node.lock_ref += 1 
   return delta
```

### API

- Cache when request finished or unfished
- Evcit

```python
 #######################
 # Cache Unfinished Req
  #######################
  def cache_unfinished_req(req):
    token_ids = req.fill_ids
    
    # get `out_cache_loc`, which is actually Value
    kv_indices = req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
      ]
      
      if page_size != 1:
        page_aligned_len = len(kv_indices) // page_size * page_size
        # V align
          page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
      else:
          page_aligned_len = len(kv_indices)
          page_aligned_kv_indices = kv_indices.clone()
      
      # K align
      page_aligned_token_ids = token_ids[:page_aligned_len]

      # insert K,V
      new_prefix_len = insert(page_aligned_token_ids, page_aligned_kv_indices)
      
      # remove repetive part
      token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
      )
      
      #  get prefixed `out_cache_loc` and `new_last_node`
      new_indices, new_last_node = self.match_prefix(page_aligned_token_ids)
      
      # only write new `out_cache_loc`
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
 # Cache Finished Req
 #####################
  def cache_finished_req(req):
   if self.disable:
     # if disable radix tree, free the KV Cache of this finished req directly
     
     # get `out_cache_loc`
     kv_indices = req_to_token_pool.req_to_token[
              req.req_pool_idx, : len(req.origin_input_ids) + len(req.output_ids) - 1
          ]
          
          # free `req slots` and `token_to_kv_pool slots`
          token_to_kv_pool_allocator.free(kv_indices)
          req_to_token_pool.free(req.req_pool_idx)
          return
          
     # if using radix tree, don't free KV Cache instantly for reusing opportunities
     
     # get token_ids, which is actually key
     token_ids = (req.origin_input_ids + req.output_ids)[:-1]
     
     # get `out_cache_loc`, which is actually value
     kv_indices = req_to_token_pool.req_to_token[
        req.req_pool_idx, : len(token_ids)
    ]
    
    # assuming page size is 1, so it is automatically aligned
    page_aligned_len = len(kv_indices)
     page_aligned_kv_indices = kv_indices.clone()
    
    # insert the [token_ids, out_cache_loc] into radix tree for reuse
    new_prefix_len = insert(
         token_ids[:page_aligned_len], page_aligned_kv_indices
    )
    
     # only free [len(prefix_indices): new_prefix_len] part of kv pool, why?
     # since these part of `out_cache_loc` are REPETITIVE (REDUNDANT)!
     
     # The whole process is as follows:
     # `req.prefix_indices` is computed when it is scheduled at first
     # `new_prefix_len` is the prefix lens when it is finished
     # [len(req.prefix_indices): new_prefix_len] is the repetive part during which computed 
    token_to_kv_pool_allocator.free(
          kv_indices[len(req.prefix_indices) : new_prefix_len]
     )
     
     # free `req slot` for sure 
     # since the req has been finished, its req_pool_idx can be used for other reqs
     req_to_token_pool.free(req.req_pool_idx)
     
     # dec lock_ref of those node owns out_cache_loc[:len(prefix_indices)]
     # these part will be possibly evictable 
     # but Note: these `out_cache_loc` have not been evicted yet
     dec_lock_ref(req.last_node)
```

```python
  def evict(num_tokens: int):
    if disable: return
    
    leaves = _collect_leaves()
    
    # sort by `last_access_time` (LRU)
    heapq.heapify(leaves)
    
    num_evicted = 0
    while num_evicted < num_tokens and len(leaves):
      x = heapq.heappop(leaves)
      if x == self.root_node: break
      
      # if some reqs are pointing to this node, skip it
            if x.lock_ref > 0: continue
             
            # free this node's `out_cache_loc`
            token_to_kv_pool_allocator.free(x.value)
            
            num_evicted += len(x.value)
            _delete_leaf(x)
            
            # add new leaves node for next evitable
            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)
            
  def _delete_leaf(node):
  
    # delete this node from its parent
    for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        
        # update evicatble_size
        evictable_size_ -= len(node.key)
    
```

## Usage

How to use the above API provided by `radix_cache_tree` ?

### Cache

When `prefill` is over,

```python
def process_batch_result_prefill(batch, result):
  for i, (req, next_token_id) in enumerate(batch.reqs, result.next_token_ids):
    req.output_ids.append(next_token_id)
        req.check_finished()
        
        if req.finished():
          tree_cache.cache_finished_req(req)
         
       elif not batch.decoding_reqs or req not in batch.decoding_reqs:
            # This updates radix so others can match
            tree_cache.cache_unfinished_req(req)
```

When `decode`  is over,

```python
def process_batch_result_decode(batch, result):
  for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
    req.check_finished()
    
    if req.finished():
           tree_cache.cache_finished_req(req)
```

<aside>
💡

Only when `decode` finished, tree_cache cached its (`token_ids, out_cache_loc` )

</aside>

### Evict

Evict, which is also free `out_cache_loc` , happened when available_size in `token_to_kv_pool` cannot support the incoming req

```python
def alloc_token_slots(num_tokens: int, backup_state: bool = False):
    if token_to_kv_pool_allocator.available_size() < num_tokens:
      if tree_cache is not None:
          tree_cache.evict(num_tokens)
          
  out_cache_loc = token_to_kv_pool_allocator.alloc(num_tokens)
```

# Reference

- <https://hebiao064.github.io/fa3-attn-backend-basic>
