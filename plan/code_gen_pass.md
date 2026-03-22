你帮我写一个 pass 用于代码生成,有一些信息如下:
- 我要操作的 graph 是经过了 nandmachine/frontend/core/passes/normalize.py 中 normalized pass 的 NxGraph，其中的节点已经洗过一次了，部分的 node 的 meta 中存在 hook module 这样的参数。
- 初始化一个总的 macro op list:list[MacroOp]
- 按照拓扑序的节奏，开始遍历每一个节点，对于每一个拥有 hook module 的节点，进行如下的操作:
    - 调用这个 hook module 里面的 macro_code_gen, 传入其需要的参数 NxGraphMeta，从 Graph的 meta 中拿一下
    - 将 hook module 生成的 macro op list 放入到 总的 list 中
    - 运行的时候，输出一下当前处理的 node 和 生成的 macro op 的数量
- 最后在 graph 的 meta 中记录一个 完整的 macro_op_list 

