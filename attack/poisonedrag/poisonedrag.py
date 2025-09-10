def poisonedrag_formatter(CoT_exmaple_set, prefix=""):
    """
    Insert the trigger tokens into the CoT examples
    """
    CoT_prefix = prefix
    trigger_sequence = ""
    for idx, example in enumerate(CoT_exmaple_set):
        if "NOTICE" in example:
            example = example.format(trigger = trigger_sequence, action = "SUDDEN STOP")
            example = "Here is a RAG question." + example
        
        CoT_prefix += example
    
    CoT_prefix += "\n"
    
    return CoT_prefix, ""