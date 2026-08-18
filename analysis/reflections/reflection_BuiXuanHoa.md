# Individual Reflection — Lab 18

**Tên:** Bùi Xuân Hòa  
**Module phụ trách:** Toàn bộ Pipeline (M1, M2, M3, M4, M5)

---

## 1. Đóng góp kỹ thuật

- Module đã implement:
  - **M1 (Chunking):** Semantic chunking, Hierarchical chunking (parent-child), Structure-aware chunking.
  - **M2 (Hybrid Search):** BM25 tiếng Việt (underthesea tokenization) + Dense vector (Qdrant) + RRF (Reciprocal Rank Fusion).
  - **M3 (Reranking):** Cross-Encoder reranking top-20 xuống top-3, benchmark latency.
  - **M4 (Evaluation):** RAGAS evaluation (faithfulness, answer_relevancy, context_precision, context_recall), failure analysis theo Diagnostic Tree.
  - **M5 (Enrichment):** Chunk summarization, HyQA generation, Contextual prepending, Auto metadata extraction.
- Các hàm/class chính đã viết: `chunk_semantic`, `chunk_hierarchical`, `chunk_structure_aware`, `BM25Search`, `DenseSearch`, `HybridSearch`, `CrossEncoderReranker`, `evaluate_ragas`, `failure_analysis`, `enrich_chunks`.
- Số tests pass: 37 / 37 unit tests (100% pass).

## 2. Kiến thức học được

- Khái niệm mới nhất: Parent-Child hierarchical chunking giúp cân bằng giữa độ chính xác khi tìm kiếm (child chunk nhỏ) và độ đầy đủ ngữ cảnh khi sinh câu trả lời (parent chunk lớn).
- Điều bất ngờ nhất: RRF kết hợp BM25 tiếng Việt và Dense vector cải thiện đáng kể retrieval recall so với tìm kiếm dense đơn thuần.
- Kết nối với bài giảng: Áp dụng toàn bộ kiến trúc Advanced/Production RAG pipeline từ indexing, retrieval, reranking đến evaluation framework.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Xung đột bộ nhớ C++ trên Windows khi nạp mô hình qua HuggingFace/PyTorch safetensors và lỗi kết nối Qdrant server khi chưa bật Docker.
- Cách giải quyết: Cấu hình `automodel_args={"low_cpu_mem_usage": False}`, dùng mô hình phù hợp và xây dựng cơ chế tự động fallback sang Embedded Qdrant storage. Tích hợp Groq API để chạy evaluation và enrichment miễn phí tốc độ cao.
- Thời gian debug: ~45 phút.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Thử nghiệm thêm ColBERT / Late Interaction reranking và tối ưu hóa async embedding cho pipeline.
- Module nào muốn thử tiếp: Thử nghiệm thêm Contextual Retrieval nâng cao với GraphRAG.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5/5 |
| Code quality | 5/5 |
| Teamwork | 5/5 |
| Problem solving | 5/5 |
