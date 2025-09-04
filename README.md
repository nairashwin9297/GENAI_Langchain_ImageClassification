# LangChain RAG & CNN CIFAR-10 Classification

Advanced ML implementations: LangChain RAG system with input/output parsing and systematic CNN architecture study for CIFAR-10 image classification.

## Quick Start

```bash
# Install dependencies
pip install langchain faiss-cpu sentence-transformers tensorflow

# Part A: Run RAG system
python rag_demo.py "What is Boston known for?"

# Part B: Run CNN experiments  
python cnn_experiments.py
```

## Part A: LangChain RAG System

**Complete RAG pipeline with structured output parsing**

### Core Implementation
- **RAG Pipeline**: Document loading → Text chunking → Vector embeddings → FAISS retrieval → Generation
- **Input Parsing**: Text validation, cleaning, format normalization
- **Output Parsing**: Pydantic models for structured JSON responses
- **Free Stack**: SentenceTransformers + FAISS + Flan-T5 (no API costs)

### Key Results
- **Success Rate**: 75% on diverse test inputs
- **Input Handling**: Automatic whitespace cleaning, question formatting
- **Output Formats**: JSON, human-readable, confidence scoring
- **Error Handling**: Graceful validation with meaningful error messages

<details>
<summary>Technical Details</summary>

#### Pydantic Output Schema
```python
class QAResponse(BaseModel):
    answer: str = Field(..., description="The answer")
    sources: List[str] = Field(..., description="Source docs")
    confidence: str = Field(..., description="Confidence level") 
    word_count: int = Field(..., description="Answer length")
```

#### Architecture Components
- **Document Layer**: DirectoryLoader + RecursiveCharacterTextSplitter
- **Vector Layer**: SentenceTransformers (all-MiniLM-L6-v2) + FAISS
- **Generation Layer**: HuggingFace Flan-T5-Small + LangChain RetrievalQA
- **Parsing Layer**: PydanticOutputParser with fallback handling

</details>

## Part B: CNN Architecture Study

**Systematic analysis of CNN design choices on CIFAR-10**

### Best Configuration
- **Architecture**: 3 blocks + BatchNorm + Dropout + ReLU
- **Test Accuracy**: 74.4%
- **Key Finding**: Dropout essential for preventing overfitting

### Experimental Design
- **Grid Search**: 16 configurations (2×2×2×2 parameter combinations)
- **Parameters**: Network depth, batch normalization, dropout, activation functions
- **Training**: Early stopping, Adam optimizer, 10% validation split

### Critical Findings

| Finding | Impact | Evidence |
|---------|---------|----------|
| **Dropout Essential** | +5.1% improvement | Models without dropout consistently underperformed |
| **Depth Matters** | +3.0% average boost | 3-block networks outperformed 2-block |
| **ReLU Superior** | +1.7% over Tanh | Consistent across all configurations |
| **BatchNorm Risk** | Severe overfitting | Without dropout: worst performance (61-64%) |

<details>
<summary>Complete Results Table</summary>

| Rank | Blocks | BatchNorm | Dropout | Activation | Accuracy |
|------|--------|-----------|---------|------------|----------|
| 1 | 3 | ✓ | ✓ | ReLU | **74.40%** |
| 2 | 3 | ✗ | ✓ | ReLU | 73.97% |
| 3 | 3 | ✗ | ✓ | Tanh | 71.89% |
| 4 | 2 | ✓ | ✓ | ReLU | 71.51% |
| 5 | 3 | ✗ | ✗ | ReLU | 71.39% |

*Full results: 16 configurations tested*

</details>

## Installation

```bash
pip install langchain faiss-cpu sentence-transformers transformers tensorflow pandas pydantic
```


## Key Takeaways

**RAG System**: Production-ready pipeline with robust input/output parsing demonstrates advanced LangChain capabilities using entirely free components.

**CNN Study**: Systematic experimentation reveals critical importance of regularization balance - dropout prevents overfitting while batch normalization alone can harm performance.

---

*Technologies: LangChain, FAISS, SentenceTransformers, TensorFlow, Pydantic*
