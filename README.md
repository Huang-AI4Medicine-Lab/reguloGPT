# reguloGPT: Harnessing GPT for Knowledge Graph Construction of Molecular Regulatory Pathways
# Abstract
Motivation: Molecular Regulatory Pathways (MRPs) are crucial for understanding biological functions.
Knowledge Graphs (KGs) have become vital in organizing and analyzing MRPs, providing structured
representations of complex biological interactions. Current tools for mining KGs from biomedical literature are
inadequate in capturing complex, hierarchical relationships and contextual information about MRPs. Large
Language Models (LLMs) like GPT-4 offer a promising solution, with advanced capabilities to decipher
the intricate nuances of language. However, their potential for end-to-end KG construction, particularly for
MRPs, remains largely unexplored.  

Results: We present reguloGPT, a novel GPT-4 based in-context learning prompt, designed for the end-to-
end joint name entity recognition, N-ary relationship extraction, and context predictions from a sentence that
describes regulatory interactions with MRPs. Our reguloGPT approach introduces a context-aware relational
graph that effectively embodies the hierarchical structure of MRPs and resolves semantic inconsistencies
by embedding context directly within relational edges. We created a benchmark dataset including 400
annotated PubMed titles on N6-methyladenosine (m6A) regulations. Rigorous evaluation of reguloGPT on
the benchmark dataset demonstrated marked improvement over existing algorithms. We further developed
a novel G-Eval scheme, leveraging GPT-4 for annotation-free performance evaluation and demonstrated its
agreement with traditional annotation-based evaluations. Utilizing reguloGPT predictions on m6A-related
titles, we constructed the m6A-KG and demonstrated its utility in elucidating m6A’s regulatory mechanisms
in cancer phenotypes across various cancers. These results underscore reguloGPT’s transformative potential
for extracting biological knowledge from the literature.  

  Availability and implementation: The source code of reguloGPT, the m6A title and benchmark datasets,
and m6A-KG are available at: https://github.com/Huang-AI4Medicine-Lab/reguloGPT.  

  Key words: Molecular Regulatory Pathways, Knowledge Graph, GPT, In Context Learning, m6A mRNA Methylation

# Prompt
![prompt_figure](https://github.com/Huang-AI4Medicine-Lab/reguloGPT/assets/69179826/643468a4-7cd3-4e18-a4d5-d17047075018)
(A) Baseline prompt including instruction, definition, and output format. (B) Demonstration in few-shot
prompt. (C) Demonstration in CoT prompt
# Knowledge Graph:
![m6A-KG](https://github.com/Huang-AI4Medicine-Lab/reguloGPT/assets/69179826/5ee5300d-ba37-47cd-9517-de2e7b3526f9)
Cancer-type specific KG of (A) Breast cancer, (B) Myeloidleukemia, and (C) Lung cancer. Extracted pathways are shown to
the left. Edge colors are associated with the supporting titles.
