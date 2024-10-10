mermaid '''
flowchart LR
subgraph Query Embeddings
A(User) --> B(Question)
B --> C(LLM Generative AI)
C --> D(Query Embeddings)
D --> E(Semantic Search)
E --> F(Ranked Search)
F --> C
C --> A
end

    subgraph PreProcess
        G(PDFs) --> H(Extract data/context)
        H --> I(Split in chunks)
    end

    subgraph Knowledge Base
        I --> J1(Text chunk 1) --> K1(Embeddings 1)
        I --> J2(Text chunk 2) --> K2(Embeddings 2)
        I --> J3(Text chunk 3) --> K3(Embeddings 3)
        I --> J4(Text chunk 10) --> K4(Embeddings 10)
        K1 --> L(Build Semantic Index)
        K2 --> L(Build Semantic Index)
        K3 --> L(Build Semantic Index)
        K4 --> L(Build Semantic Index)
        L --> M(Knowledge Base)
    end

    E --> M
    M --> F

'''
