# **ArXivCode: Bridging Theory and Implementation in AI Research**

## **Project Proposal**  **Course:** Introduction to LLM based Generative AI Systems (COMSE6998-015) – Fall 2025 **Instructors:** Dr. Parijat Dube and Dr. Chen Wang **Date:** October 20, 2025 **Project Team:** [Nicholas Julian Bindela](mailto:njb2163@columbia.edu)	UNI: njb2163 [Pranati Modumudi](mailto:pm3361@columbia.edu) 	UNI: pm3361 [Tomas Pasiecznik](mailto:tp2758@columbia.edu)		UNI: tp2758

---

## **1\. Problem Statement and Objectives**

### **The Problem**

The gap between theoretical AI research and practical implementation creates significant barriers for researchers and practitioners. While ArXiv hosts cutting-edge papers and GitHub contains implementations, discovering and understanding how theoretical concepts translate to actual code remains time-consuming and challenging. This disconnect impedes research reproducibility, learning, and practical adoption of new methods.

### **Research Questions**

1. How can fine-tuned language models effectively align theoretical concepts in papers with code implementations?  
2. What retrieval architecture enables accurate mapping between a paper and its relevant code snippets?  
3. What methods can be used to ensure that model-generated code suggestions preserve functional correctness while adapting to user-specific constraints?

### **Project Objectives**

Develop **ArXivCode**, a dual-model research assistant that:

* Fine-tunes a code understanding model (CodeBERT/StarCoder) and language model (LLaMA/Mistral) on paired ArXiv papers and GitHub repositories  
* Enables users to query theoretical concepts and receive relevant code snippets with explanatory annotations  
* Provides code modification suggestions adapted to user requirements

**Target Deliverables**:

* Curated dataset of 500 ArXiv-GitHub pairs with fine-grained alignments  
* Two fine-tuned models with documented training procedures  
* Interactive retrieval system with web-based demonstration

---

## **2\. Methodology and Technical Approach**

### **Data Collection**

* Source paper-code pairs from Papers With Code API and GitHub search  
* Filter for quality (\>50 GitHub stars, 2020-2025 publications)  
* Focus on cs.CL (Computational Linguistics) and cs.LG (Machine Learning) domains (Python implementations only)

### **Model Architecture**

**Component 1: Code Understanding Model**

* Base: CodeBERT or StarCoder-Base (150M-3B parameters; current options)  
* Training: Contrastive learning on paper descriptions and corresponding code (InfoNCE loss)  
* Output: Dense embeddings for semantic code retrieval

**Component 2: Paper Comprehension Model**

* Base: LLaMA-3-8B or Mistral-7B  
* Training: Instruction fine-tuning with LoRA/QLoRA for parameter efficiency  
* Task: Generate code references and explanations given paper queries

**Component 3: Retrieval System**

* Dense retrieval using code embeddings  
* Cross-encoder re-ranking for relevance  
* Context-aware snippet extraction and annotation

### **Training Strategy**

1. **Phase 1**: Contrastive pre-training of code encoder on paper-code pairs  
2. **Phase 2**: Instruction fine-tuning of language model on query-response pairs  
3. **Phase 3**: Retrieval optimization for precision and recall

---

## **3\. Innovation and Relevance**

### **Novel Contributions**

* **Dual-model specialization**: First system fine-tuned specifically on paper-code pairs for research assistance  
* **Fine-grained alignment**: Paragraph-to-function level mapping beyond repository linking  
* **Contextual adaptation**: Code modification suggestions based on user requirements

### **Alignment with GenAI Trends**

* Multimodal understanding (text and code)  
* Retrieval-augmented generation  
* Specialized domain fine-tuning  
* Tool-augmented LLMs

### **Impact on the Field**

* Accelerates research-to-implementation from hours to minutes  
* Improves research reproducibility and accessibility  
* Democratizes access to cutting-edge AI implementations

---

## **4\. Feasibility and Timeline**

### **8-Week Timeline**

* **Weeks 1-2**: Data collection and preprocessing (200 pairs for prototyping)  
* **Weeks 3-4**: Model fine-tuning and initial retrieval system  
* **Weeks 5-6**: System integration, optimization, and end-to-end pipeline  
* **Week 7**: Front-end development, integration with API  
* **Week 8**: Final report, tutorial, and presentation preparation

### **Resource Requirements**

* **Compute**: 1x A100 (40GB) or 2x A6000 (48GB) for \~50 GPU hours total (Columbia CS GPU cluster or Colab Pro+)  
* **Data**: Papers With Code API, ArXiv bulk data, GitHub API (all free/open access)  
* **Software**: PyTorch, Transformers, FAISS, LangChain (all open-source)

### **Risk Mitigation**

| Risk | Mitigation | Fallback |
| :---- | :---- | :---- |
| Data quality issues | Filter by popularity and citations | Manual curation of 500 high-quality pairs |
| Training complexity | Use LoRA and smaller models (1B-3B) | Focus on retrieval with frozen embeddings |
| Evaluation subjectivity | Clear annotation guidelines | Emphasize quantitative metrics |
| Scope creep | Define MVP: retrieval with 500 papers | Defer code modification features |

### **Scope Boundaries**

**In Scope**: ArXiv CS papers (2020-2025), Python code, retrieval and annotation  
**Out of Scope**: Multi-language support, code execution, automated debugging, paper-to-code generation

---

## **5\. Differentiation from Existing Solutions**

* **Papers With Code**: Repository-level linking only, no intelligent retrieval  
* **GitHub Search**: Keyword-based, no paper context  
* **CodeBERT**: General-purpose, not research-specialized  
* **GPT-4**: Not optimized for paper-code alignment, no specialized training

**ArXivCode's Advantage**: Specialized training on research paper-code pairs with fine-grained, semantically-aware retrieval optimized for academic workflows.

---

## **Conclusion**

ArXivCode addresses a critical gap in research workflows through novel application of fine-tuned LLMs for paper-code alignment. The project is technically feasible, practically impactful, and well-aligned with course objectives in GenAI system development. By combining technical innovation with clear practical utility, ArXivCode will accelerate research reproducibility and democratize access to cutting-edge AI implementations.