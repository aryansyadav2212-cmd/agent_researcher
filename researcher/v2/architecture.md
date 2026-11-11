## structure_index.py
- get_node("node_002")

> map each node to its parent_id , it allows us to navigate upward
- get_parent("node_002")

>retrives blocks by key
- get_block("node_id" , "block_id")

> maps each node to its ordered content blocks
- get_blocks("node_id")

this connects to contex_expander.py



## context_expander.py
takes the retrived evidence block from structure_index.py and finds the sourrounding block


## evidence_evaluator.py
evidence evaluation
seperating evaluation in three diamenstions :- Relevance , Sufficiency , Coverage
> remove duplicate blocks



