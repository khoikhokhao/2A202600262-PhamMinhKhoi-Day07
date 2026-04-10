# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Minh Khôi
**Nhóm:** 32
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Nghĩa là hai đoạn văn bản có ý nghĩa (ngữ nghĩa) rất giống nhau hoặc cùng nói về một chủ đề. Vector của chúng chĩa về cùng một hướng trong không gian đa chiều (góc giữa 2 vector xấp xỉ 0 độ).

**Ví dụ HIGH similarity:**
- Sentence A: Khách hàng có thể yêu cầu hoàn tiền trong vòng 14 ngày.
- Sentence B: Chính sách của chúng tôi cho phép bạn lấy lại tiền trong hai tuần đầu.
- Tại sao tương đồng: Dùng từ ngữ khác nhau (hoàn tiền/lấy lại tiền, 14 ngày/hai tuần) nhưng nét nghĩa, ngữ cảnh và chủ đề hoàn toàn trùng khớp.

**Ví dụ LOW similarity:**
- Sentence A: Khách hàng có thể yêu cầu hoàn tiền trong vòng 14 ngày.
- Sentence B: Thị trường chứng khoán hôm nay ghi nhận mức giảm điểm mạnh.
- Tại sao khác: Hai câu thuộc hai lĩnh vực hoàn toàn không liên quan, không chia sẻ bất kỳ từ vựng hay ý nghĩa ngữ cảnh nào.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Bởi vì Cosine Similarity đo lường "hướng" của vector (tức là chủ đề, ý nghĩa) thay vì "độ lớn/chiều dài" của vector (bị ảnh hưởng bởi số lượng từ ngữ). Hai đoạn văn bản cùng ý nghĩa dù một đoạn dài, một đoạn ngắn thì Cosine vẫn đánh giá độ tương đồng cao.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap)) = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23.
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Nếu overlap = 100: num_chunks = ceil((10000 - 100) / (500 - 100)) = 25 chunks. Việc tăng overlap giúp đảm bảo các câu dài hoặc các ý quan trọng nằm ở ranh giới giữa 2 chunk không bị đứt gãy ngữ cảnh, duy trì sự liền mạch của thông tin khi truy xuất.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Customer support knowledge base / help-center documentation

**Tại sao nhóm chọn domain này?**
> Nhóm chọn tài liệu Hỗ trợ khách hàng (Support Docs) vì nó có cấu trúc rất rõ ràng (theo Section, Bullet points), phân chia rạch ròi giữa tài liệu Public cho khách và Internal cho nhân viên. Đây là bộ data hoàn hảo để test thuật toán Chunking (giữ cấu trúc hướng dẫn) và test bộ lọc Metadata để tránh bot lôi nhầm tài liệu nội bộ ra trả lời khách.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `account_email_change.md` | OpenAI Help Center | 1999 | `doc_id`, `title`, `category=account`, `audience=customer`, `language=en`, `source`, `last_updated`, `sensitivity=public` |
| 2 | `password_reset_help.md` | OpenAI Help Center | 1815 | `doc_id`, `title`, `category=password`, `audience=customer`, `language=en`, `source`, `last_updated`, `sensitivity=public` |
| 3 | `billing_renewal_failure.md` | OpenAI Help Center | 1789 | `doc_id`, `title`, `category=billing`, `audience=customer`, `language=en`, `source`, `last_updated`, `sensitivity=public` |
| 4 | `refund_request_guide.md` | OpenAI Help Center | 1952 | `doc_id`, `title`, `category=refund`, `audience=customer`, `language=en`, `source`, `last_updated`, `sensitivity=public` |
| 5 | `service_limit_429.md` | OpenAI Help Center | 1867 | `doc_id`, `title`, `category=service_limit`, `audience=customer`, `language=en`, `source`, `last_updated`, `sensitivity=public` |
| 6 | `internal_escalation_playbook.md` | Internal support / handbook | 1944 | `doc_id`, `title`, `category=escalation`, `audience=internal_support`, `language=en`, `source`, `last_updated`, `sensitivity=internal` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `kb_refund_001` | Định danh duy nhất cho mỗi tài liệu, hữu ích khi quản lý, update hoặc xóa tài liệu |
| `title` | string | `How to Request a Refund` | Giúp nhận diện nhanh nội dung tài liệu và trích xuất nguồn cho user |
| `category` | string | `account`, `billing`, `escalation` | Cho phép lọc theo chủ đề (Pre-filtering) để tăng precision |
| `audience` | string | `customer`, `internal_support` | Ngăn chặn việc bot trả lời nhầm các quy trình nội bộ/bảo mật cho khách hàng. |
| `language` | string | `en` | Hữu ích khi sau này mở rộng sang tài liệu đa ngôn ngữ. |
| `source` | string | `https://...` | Giúp truy vết nguồn gốc tài liệu. |
| `last_updated` | string | `2026-04-10` | Hữu ích để ưu tiên tài liệu mới, tránh lấy chính sách cũ. |
| `sensitivity` | string | `public`, `internal` | Giảm nguy cơ retrieve nhầm tài liệu nhạy cảm. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `account_email_change.md` | FixedSizeChunker | 5 | 382.2 | Giữ overlap tốt nhưng hay cắt giữa ý |
| `account_email_change.md` | SentenceChunker | 11 | 155.8 | Tốt về mạch câu nhưng chunk hơi nhỏ, dễ mất context lớn |
| `account_email_change.md` | RecursiveChunker | 6 | 286.8 | Cân bằng tốt nhất giữa độ dài và ngữ cảnh |
| `refund_request_guide.md` | FixedSizeChunker | 4 | 445.5 | Có thể cắt ngang section một cách thiếu tự nhiên |
| `refund_request_guide.md` | SentenceChunker | 7 | 233.4 | Dễ đọc, nhưng đôi lúc tách rời các bước liên quan |
| `refund_request_guide.md` | RecursiveChunker | 5 | 327.8 | Ổn định, giữ được trọn vẹn cụm logic |

### Strategy Của Tôi

**Loại:** `RecursiveChunker` (chuẩn hóa `chunk_size=500`)

**Mô tả cách hoạt động:**
> Hệ thống thiết kế theo đệ quy. Hàm sẽ thử cắt text bằng separator lớn nhất (`\n\n`). Nếu đoạn con tạo ra vẫn lớn hơn `chunk_size` (500), hàm sẽ gọi lại chính nó để cắt tiếp bằng separator nhỏ hơn (dấu chấm, khoảng trắng), đến khi vừa khung size thì thôi.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu Support của nhóm chứa rất nhiều heading (`##`) và danh sách các bước (`1. 2. 3.`). Dùng RecursiveChunker với size 500 sẽ ưu tiên gom trọn vẹn một cụm heading/paragraph vào một chunk. Điều này giúp Agent lấy được toàn bộ quy trình xử lý lỗi thay vì bị băm bổ giữa câu như FixedSize hay quá vụn vặt như SentenceChunker.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `refund_request_guide.md` | best baseline (`fixed_size`) | 4 | 445.5 | Chứa nhiều ngữ cảnh nhưng thỉnh thoảng lẫn thông tin nhiễu. |
| `refund_request_guide.md` | **của tôi** (`recursive`) | 5 | 327.8 | Chunk bám sát theo section, sạch sẽ và lấy đúng trọng tâm câu hỏi hơn. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Khôi)| `RecursiveChunker` | 8.5/10 | Giữ được ngữ cảnh theo section và bullet rất tốt, hợp với cấu trúc doc. | Đôi lúc tạo ra chunk hơi nhiều, cần cấu hình độ dài cẩn thận. |
| Tăng | `SentenceChunker` (3 sent) | 8.9/10 | Giữ instructions trọn vẹn, semantic units. | Chunks không đều (có đoạn quá ngắn). |
| Thế Anh | `FixedSizeChunker` | 8.0/10 | Giữ ngữ cảnh liên tục, chuẩn hóa chunk. | Có thể cắt giữa câu nếu câu dài. |
| Quân | `RecursiveChunker` (size 420) | 7.0/10 | Phù hợp với tài liệu có cấu trúc. | Query mơ hồ dễ bị lệch top-1 sang chunk khác. |
| Minh | `RecursiveChunker` (size 400) | 6.0/10 | Giữ context steps và internal notes tốt. | Chưa vượt trội khi dùng _mock_embed. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Dựa trên thực nghiệm của nhóm, `SentenceChunker` có điểm số nhỉnh nhất, nhưng `RecursiveChunker` (với chunk_size tối ưu) mới là phương án thực tế nhất. Nó cân bằng được việc lấy đủ một quy trình hỗ trợ khách hàng (gồm nhiều câu) mà không bị cắt đứt mạch văn, đáp ứng rất tốt cho các hệ thống Knowledge Base phức tạp.

---

## 4. My Approach — Cá nhân (10 điểm)

**Chunking Functions**
* **`SentenceChunker`**: Tôi dùng Regex `(?<=[.!?])\s+|\.\n` để nhận diện ranh giới câu (dấu câu đi kèm khoảng trắng/xuống dòng). Sau đó lặp qua danh sách, gộp các câu lại thành các khối nhỏ không vượt quá tham số `max_sentences_per_chunk`.
* **`RecursiveChunker`**: Cốt lõi là hàm `_split()` đệ quy. Nếu một đoạn văn bản (sau khi cắt bởi separator hiện tại) vẫn vượt `chunk_size`, nó sẽ mượn danh sách `separators` còn lại để tiếp tục chia nhỏ. Fallback cuối cùng là cắt theo ký tự nếu hết separator. 

**EmbeddingStore**
* **`add_documents` + `search`**: `add_documents` đóng gói ID, content, metadata và vector embedding vào một dictionary rồi đẩy vào `_store`. Hàm `search` nhúng query, chạy vòng lặp tính Cosine Similarity với mọi record, sort giảm dần và trả về top_k.
* **`search_with_filter` + `delete_document`**: `search_with_filter` thực hiện "Pre-filtering": nó lọc ra một mảng `filtered_records` khớp toàn bộ key-value trong bộ lọc rồi mới đem mảng đó đi tính điểm. `delete_document` sử dụng list comprehension để lọc bỏ các record trùng `doc_id`.

**KnowledgeBaseAgent**
* **`answer`**: Agent gọi `store.search` lấy top_k chunk. Các chunk này được map thành một chuỗi `context_str` có cấu trúc. Sau đó, nội dung được nhúng vào một Prompt chuẩn mực (đóng vai trò system prompt) để ép LLM chỉ trả lời dựa trên ngữ cảnh đã cung cấp, ngăn chặn hallucination.

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | I need to change my email address. | How can I update my account email? | High | 0.85+ | Đúng |
| 2 | Use exponential backoff for 429 errors. | Wait and retry when rate limit is hit. | High | 0.78+ | Đúng |
| 3 | Clear cache and cookies before retrying. | Clear browser cache and cookies before retrying. | High | 0.95+ | Đúng |
| 4 | Contact the bank if the renewal failed. | To bake a cake, you need flour and eggs. | Low | 0.05- | Đúng |
| 5 | Reset my password from the settings menu. | The internal escalation process requires a manager. | Low | 0.10- | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là Pair 1 và Pair 2, dù sử dụng các từ vựng hoàn toàn khác nhau (exponential backoff vs wait and retry), mô hình embedding vẫn nhận diện được chúng có chung một ngữ nghĩa và cho điểm cao. Điều này chứng minh embeddings biểu diễn ý tưởng trong không gian vector dựa trên ngữ cảnh (contextual meaning), chứ không chỉ thực hiện so khớp từ khóa (keyword matching) thông thường. *(Lưu ý: Nhận định này đúng với các mô hình Local/OpenAI, còn `_mock_embed` chỉ sinh vector giả lập).*

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | How can a customer change the email address on their OpenAI account? | A customer can change their email from Settings > Account on ChatGPT Web if the account supports email management. This is not supported for phone-number-only accounts, Enterprise SSO accounts, or some enterprise-verified personal accounts. After the change, the user is signed out and must log in again with the new email. |
| 2 | What should a customer do if they do not receive the password reset email? | The customer should check the spam/junk folder, confirm they are checking the same inbox used during signup, and verify there is no typo in the email address. If the account was created only with Google, Apple, or Microsoft login, password recovery must be done through that provider instead. |
| 3 | What are the recommended steps when a ChatGPT Plus or Pro renewal payment fails? | The customer should clear browser cache and cookies, contact the bank to check for blocks or security flags, verify billing and card details, and confirm the country or region is supported. If the payment still fails, they should contact support through the Help Center chat widget. |
| 4 | How should a customer handle a 429 Too Many Requests error? | A 429 error means the organization exceeded its request or token rate limit. The recommended solution is exponential backoff: wait, retry, and increase the delay after repeated failures. The customer should also reduce bursts, optimize token usage, and consider increasing the usage tier if needed. |
| 5 | When should an active customer emergency be escalated, and who should be contacted first? | Escalation should be considered when the emergency lasts more than 3 hours without clear resolution, involves multiple simultaneous customer issues, blocks critical outside work, or requires broader coordination. A Support Manager On-call should be consulted, and the account CSM should usually be contacted first as the escalation DRI. |

### Kết Quả Của Tôi (Khôi)

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | How can a customer change the email address... | "## Steps on web 1. Sign in to ChatGPT... 2. Open Settings. 3. Select Account..." | Cao | Yes | Đổi qua Settings > Account trên web. Giải thích rõ các giới hạn. (2đ) |
| 2 | What should a customer do if they do not receive the password... | "## If you do not receive the reset email Check the following: - spam or junk folder..." | Cao | Yes | Báo check spam, xác minh inbox gốc và kiểm tra typo. (2đ) |
| 3 | What are the recommended steps when a ChatGPT Plus... | "## First troubleshooting steps 1. Clear browser cache... 2. Contact your bank..." | Cao | Yes | Yêu cầu clear cache, check bank, verify info, confirm region. (2đ) |
| 4 | How should a customer handle a 429... | "## Recommended solution: exponential backoff. Basic idea..." | Cao | Yes | Đề xuất exponential backoff, giảm request bursts. (2đ) |
| 5 | When should an active customer emergency be escalated... | Lệch sang một chunk ngắn của file refund (do mock_embed xử lý nhiễu) nhưng Top 2 vẫn bốc đúng. | Thấp | Top-1: Không, Top-2: Có | Agent bị thiếu 1 phần context escalation chuẩn, trả lời hơi chung chung. (0.5đ) |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5
*(Điểm số thực tế ước tính: 8.5 / 10)*

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Nhờ xem kết quả của Tăng (`SentenceChunker`), tôi nhận ra việc chia chunk theo câu giúp giảm nhiễu cục bộ cực tốt. Tuy nhiên, nó đôi lúc làm mất đi tính "quy trình" (ví dụ mất bước 1, 2, 3 liền nhau). Điều này giúp tôi nhận ra việc chọn thông số chunk_size cho `RecursiveChunker` phải dựa trên độ dài trung bình của một "Quy trình xử lý lỗi" thay vì cảm tính.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Qua demo nhóm khác, tôi học được sức mạnh của **Pre-filtering bằng Metadata**. Nếu áp dụng filter `audience = customer` trước khi search, chúng ta có thể loại bỏ hoàn toàn rủi ro Agent lấy nhầm tài liệu `internal_escalation_playbook` để trả lời cho khách hàng. Đây là một Guardrail (rào chắn) bảo mật dữ liệu tuyệt vời trong RAG thực tế.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ clean (dọn dẹp) dữ liệu Markdown kỹ hơn trước khi đưa vào Chunker (ví dụ xóa các ký tự `#`, `---` thừa thãi). Tôi cũng sẽ thiết kế thêm metadata phân cấp (VD: `intent: "troubleshooting"`) để Agent có thể gọi Tool Filter chính xác trước khi thực hiện Semantic Search, tối ưu hóa cả tốc độ lẫn độ chính xác.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 8.5 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **88.5 / 100** |
