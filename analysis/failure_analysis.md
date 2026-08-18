# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Nhóm Advanced RAG  
**Thành viên:** Bùi Xuân Hòa (Toàn bộ Pipeline: M1, M2, M3, M4, M5)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9500 | 0.9500 | +0.0000 |
| Answer Relevancy | 0.8800 | 0.8800 | +0.0000 |
| Context Precision | 0.9000 | 0.9000 | +0.0000 |
| Context Recall | 0.9140 | 0.8960 | -0.0180 |

## Bottom-5 Failures

### #1
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Tối thiểu 8 ký tự bao gồm chữ hoa, chữ thường, số và ký tự đặc biệt.
- **Got:** Mật khẩu theo quy định an toàn thông tin cần tối thiểu 8 ký tự.
- **Worst metric:** context_recall (0.80)
- **Error Tree:** Output đúng → Context thiếu chi tiết về định dạng ký tự đặc biệt →
- **Root cause:** M1 chunking chia nhỏ văn bản làm phân tách câu quy định độ dài và câu quy định loại ký tự.
- **Suggested fix:** Cải thiện Hierarchical chunking với parent chunk rộng hơn để giữ trọn vẹn ngữ cảnh.

### #2
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** 90 ngày (hoặc định kỳ 3 tháng).
- **Got:** Nhân viên cần thay đổi mật khẩu định kỳ mỗi 90 ngày một lần.
- **Worst metric:** context_recall (0.80)
- **Error Tree:** Output đúng → Context đúng một phần →
- **Root cause:** Tài liệu chính sách IT có nhiều phiên bản cập nhật thời hạn đổi mật khẩu.
- **Suggested fix:** Thêm metadata filter theo phiên bản tài liệu mới nhất (`version: latest`).

### #3
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Bắt buộc kích hoạt MFA cho tất cả tài khoản truy cập hệ thống nội bộ.
- **Got:** Bắt buộc kích hoạt MFA cho toàn bộ nhân viên truy cập hệ thống.
- **Worst metric:** context_recall (0.80)
- **Error Tree:** Output đúng → Context ngắn gọn →
- **Root cause:** BM25 keyword matching ưu tiên từ khóa "MFA" nhưng chunk chỉ chứa thông tin vắn tắt.
- **Suggested fix:** Áp dụng HyDE (Hypothetical Document Embeddings) hoặc Contextual Enrichment để bổ sung câu hỏi giả định.

### #4
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Mức lương thử việc Junior tối đa theo khung quy chế là 15 triệu/tháng (85% lương chính thức).
- **Got:** Lương thử việc Junior dao động từ 10 đến 15 triệu/tháng tùy theo đánh giá phỏng vấn.
- **Worst metric:** context_recall (0.80)
- **Error Tree:** Output đúng → Context thiếu bảng lương chi tiết →
- **Root cause:** Dữ liệu bảng lương nằm trong bảng biểu (table), chunking dạng text bị mất cấu trúc hàng/cột.
- **Suggested fix:** Áp dụng Structure-aware chunking tối ưu riêng cho bảng Markdown/HTML.

### #5
- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Expected:** Nghỉ 03 ngày nguyên lương khi kết hôn.
- **Got:** Nhân viên được nghỉ 3 ngày hưởng nguyên lương theo Bộ luật Lao động và quy chế công ty.
- **Worst metric:** answer_relevancy (0.88)
- **Error Tree:** Output đúng → Context đúng → Prompt generation sinh câu dài hơn cần thiết
- **Root cause:** Prompt template yêu cầu giải thích chi tiết dẫn đến câu trả lời dài hơn ground truth ngắn gọn.
- **Suggested fix:** Cải thiện Prompt System để trả lời súc tích, tập trung trực tiếp vào con số được hỏi.

## Case Study (cho presentation)

**Question chọn phân tích:** "Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?"

**Error Tree walkthrough:**
1. Output đúng? → Có, trả về đúng mức 15 triệu.
2. Context đúng? → Context chỉ chứa đoạn trích chính sách chung, thiếu bảng chi tiết mức lương theo cấp bậc.
3. Query rewrite OK? → BM25 bắt đúng từ khóa "Junior", "lương thử việc" nhưng dense search bị nhiễu do nhiều cấp bậc (Senior, Lead).
4. Fix ở bước: Module M1 (Structure-aware Chunking) giữ nguyên cấu trúc Table và Module M3 (Cross-Encoder) rerank ưu tiên chunk có bảng lương chi tiết.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tích hợp ColBERT / Late Interaction Reranker để tăng tốc độ rerank và cải thiện độ chính xác trên tiếng Việt.
- Tối ưu hóa parser cho tài liệu dạng Table và Form hành chính.
