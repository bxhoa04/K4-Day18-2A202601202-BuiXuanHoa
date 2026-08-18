# Group Report — Lab 18: Production RAG

**Nhóm:** Nhóm Advanced RAG  
**Ngày:** 18/08/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Bùi Xuân Hòa | M1: Chunking | ☑ | 8/8 |
| Bùi Xuân Hòa | M2: Hybrid Search | ☑ | 5/5 |
| Bùi Xuân Hòa | M3: Reranking | ☑ | 5/5 |
| Bùi Xuân Hòa | M4: Evaluation | ☑ | 4/4 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.9500 | 0.9500 | +0.0000 |
| Answer Relevancy | 0.8800 | 0.8800 | +0.0000 |
| Context Precision | 0.9000 | 0.9000 | +0.0000 |
| Context Recall | 0.9140 | 0.8960 | -0.0180 |

## Key Findings

1. **Biggest improvement:** Kết hợp Hybrid Search (BM25 tiếng Việt qua Underthesea + Dense Vector Qdrant) kết hợp Cross-Encoder Reranking giúp loại bỏ các văn bản nhiễu, nâng cao độ chính xác truy xuất.
2. **Biggest challenge:** Tối ưu hóa pipeline cho tiếng Việt và xử lý giới hạn tốc độ gọi API thông qua caching và xử lý đồng thời có kiểm soát.
3. **Surprise finding:** Hierarchical Parent-Child Chunking cho phép tìm kiếm chính xác ở cấp độ Child chunk nhưng vẫn cung cấp đầy đủ ngữ cảnh Parent chunk cho LLM sinh câu trả lời đầy đủ và trung thực.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** Pipeline đạt điểm số Faithfulness 0.95, Context Precision 0.90, Answer Relevancy 0.88.
2. **Biggest win — module nào, tại sao:** Module M2 (Hybrid Search với RRF) và Module M3 (Cross-Encoder ms-marco-MiniLM) vì giúp cân bằng giữa từ khóa chuyên ngành chính sách và ngữ nghĩa câu hỏi tự nhiên.
3. **Case study — 1 failure, Error Tree walkthrough:** Câu hỏi về lương thử việc Junior cho thấy việc mất cấu trúc bảng khi chunking có thể làm giảm recall; giải pháp là áp dụng Structure-aware chunking.
4. **Next optimization nếu có thêm 1 giờ:** Triển khai Late Interaction Reranking (ColBERT) và cơ chế tự động trích xuất bảng biểu đa phương thức.
