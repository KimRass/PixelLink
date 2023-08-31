# Research
- [TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes](https://arxiv.org/pdf/1807.01544.pdf)
- 모델의 목적 자체가 Arbitrary shaped text를 탐지하기 위함.
- 최대 라인 단위로밖에 탐지 불가하여 적합하지 않음.

# Research
- Batch size 16
```
Tried to allocate 256.00 MiB (GPU 0; 22.20 GiB total capacity; 20.28 GiB already allocated; 170.06 MiB free; 20.44 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```
- Batch size 32
```
Tried to allocate 4.00 GiB (GPU 0; 22.20 GiB total capacity; 18.23 GiB already allocated; 2.40 GiB free; 18.24 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```