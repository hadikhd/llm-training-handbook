---
id: inference and decoding
title: استنتاج و رمزگشایی
sidebar_position: 1
sidebar_label: استنتاج و رمزگشایی
---

# فصل ۱۰: استنتاج (Inference) و رمزگشایی (Decoding)
<div className="chapter-hero">

![فصل ۱۰: استنتاج و رمزگشایی](/static/img/chapters/inference.png)

</div>

[قبلی: سامانه‌ها (Systems)](./09-systems.md) |
[فهرست مطالب (Contents)](./index.md) |
[بعدی: چشم‌انداز (Outlook)](./11-outlook.md)

---

## ۱. چرا استنتاج (Inference) مهم است

آموزش (Training) یک مدل ایجاد می‌کند. استنتاج (Inference) آن مدل را به یک سامانه قابل‌استفاده تبدیل می‌کند.

در طول استنتاج (Inference)، مدل توکن‌های ورودی (Input Tokens) را دریافت می‌کند و توکن‌های خروجی (Output Tokens) را تولید می‌کند. عمل اصلی همچنان پیش‌بینی توکن بعدی (Next-Token Prediction) است، اما اولویت‌های مهندسی با مرحله آموزش (Training) تفاوت دارند.

آموزش (Training) برای یادگیری با گذردهی بالا (High Throughput) روی دسته‌های بزرگ (Large Batches) بهینه می‌شود. استنتاج (Inference) برای سرویس‌دهی به کاربران واقعی، ابزارها (Tools)، عامل‌ها (Agents)، کاربردها (Applications) و خط‌لوله‌ها (Pipelines) تحت محدودیت‌های تأخیر (Latency)، هزینه (Cost)، ایمنی (Safety) و قابلیت اطمینان (Reliability) بهینه می‌شود.

یک سامانه استنتاج (Inference System) در محیط تولید (Production) باید میان موارد زیر تعادل برقرار کند:

- کیفیت خروجی (Output Quality)
- زمان تا اولین توکن (Time to First Token)
- تأخیر بین‌توکنی (Inter-Token Latency)
- گذردهی (Throughput)
- هم‌زمانی درخواست‌ها (Request Concurrency)
- مصرف حافظه (Memory Usage)
- طول زمینه (Context Length)
- رفتار رمزگشایی (Decoding Behavior)
- کنترل‌های ایمنی (Safety Controls)
- دسترس‌پذیری (Availability)
- هزینه سرویس‌دهی (Serving Cost)

استنتاج (Inference) صرفاً اجرای کد آموزش (Training Code) در حالت ارزیابی (Evaluation Mode) نیست. این یک مسئله مستقل در سطح سامانه‌ها (Systems Problem) است.

---

## ۲. حلقه پایه تولید (Basic Generation Loop)

مدل‌های زبانی خودرگرسیو (Autoregressive Language Models) متن را هر بار یک توکن تولید می‌کنند.

در هر گام:

```text
توکن‌های ورودی
-> گذر رو‌به‌جلوی مدل (Model Forward Pass)
-> لاجیت‌ها (Logits) روی واژگان (Vocabulary)
-> قاعده رمزگشایی (Decoding Rule) توکن بعدی را انتخاب می‌کند
-> توکن انتخاب‌شده به زمینه (Context) افزوده می‌شود
-> تکرار
```
یک حلقه تولید (Generation Loop) ساده‌شده به صورت زیر است:

```python
tokens = tokenize(prompt)

while not stop_condition(tokens):
    logits = model(tokens)
    next_token = decode(logits[-1])
    tokens.append(next_token)
```
در سامانه‌های واقعی، این حلقه به‌شدت بهینه‌سازی می‌شود. مدل در هر گام کل زمینه (Context) را از ابتدا دوباره محاسبه نمی‌کند. در عوض، از حالت‌های کش‌شده توجه (Cached Attention States) استفاده می‌کند.

---

## ۳. پردازش پرامپت (Prompt Processing) و تولید توکن (Token Generation)

استنتاج (Inference) دو فاز اصلی دارد:

| فاز (Phase) | توضیح (Description) |
|---|---|
| پیش‌پرکردن (Prefill) | پردازش پرامپت ورودی (Input Prompt) و ساخت کش اولیه کلید-مقدار (Initial Key-Value Cache) |
| رمزگشایی (Decode) | تولید توکن‌های جدید، هر بار یک گام |

فاز پیش‌پرکردن (Prefill Phase) تعداد زیادی توکن پرامپت (Prompt Tokens) را به‌صورت موازی پردازش می‌کند. این فاز اغلب از نظر محاسباتی سنگین است و به یک گذر رو‌به‌جلو (Forward Pass) روی یک دسته (Batch) شباهت دارد.

فاز رمزگشایی (Decode Phase) در هر گام برای هر توالی (Sequence) یک توکن تولید می‌کند. این فاز اغلب به پهنای‌باند حافظه (Memory Bandwidth) محدود است، زیرا سامانه به‌طور مکرر وزن‌های مدل (Model Weights) و مدخل‌های کش کلید-مقدار (Key-Value Cache Entries) را می‌خواند.

این تمایز مهم است، زیرا گلوگاه‌های کارایی (Performance Bottlenecks) در این دو فاز متفاوت هستند.

```text
پرامپت بلند (Long Prompt) -> پیش‌پرکردن پرهزینه (Expensive Prefill)
پاسخ بلند (Long Answer) -> رمزگشایی پرهزینه (Expensive Decode)
کاربران هم‌زمان زیاد (Many Concurrent Users) -> فشار زمان‌بندی و حافظه (Scheduling and Memory Pressure)
```
---

## ۴. لاجیت‌ها (Logits) و احتمال‌های توکن (Token Probabilities)

مدل، لاجیت‌ها (Logits) را خروجی می‌دهد: امتیازهای نرمال‌نشده (Unnormalized Scores) روی واژگان (Vocabulary).

برای تبدیل لاجیت‌ها (Logits) به احتمال‌ها (Probabilities)، تابع سافت‌مکس (Softmax) اعمال می‌شود:

$$
p(\mathrm{token}_i) = \frac{\exp(\mathrm{logit}_i)}{\sum_j \exp(\mathrm{logit}_j)}
$$

سپس توکن بعدی را می‌توان با استفاده از راهبردهای مختلف رمزگشایی (Decoding Strategies) انتخاب کرد.

نکته مهم:

مدل مستقیماً کلمات را خروجی نمی‌دهد.  
بلکه امتیازها را روی شناسه‌های توکن (Token IDs) خروجی می‌دهد.

توکنایزر (Tokenizer) تعیین می‌کند که متن چگونه به توکن‌ها (Tokens) تبدیل شود و توکن‌های تولیدشده چگونه دوباره به متن برگردانده شوند.

---

## ۵. رمزگشایی حریصانه (Greedy Decoding)

رمزگشایی حریصانه (Greedy Decoding) در هر گام، توکنی را انتخاب می‌کند که بیشترین احتمال را دارد.


next_token = argmax(probabilities)

مزایا:

- قطعی (Deterministic)
- ساده (Simple)
- سریع (Fast)
- مفید برای خروجی‌های واقع‌محور یا مقید (Factual or Constrained Outputs)

محدودیت‌ها:

- ممکن است متن تکراری تولید کند
- ممکن است در ادامه‌های بهینه محلی (Locally Optimal Continuations) گیر کند
- ممکن است خلاقیت کمتری داشته باشد
- ممکن است وقتی بهترین پاسخ سراسری (Best Global Answer) به یک توکن میانی با احتمال کمتر نیاز دارد، شکست بخورد

رمزگشایی حریصانه (Greedy Decoding) اغلب برای وظایفی مناسب است که در آن‌ها سازگاری (Consistency) مهم‌تر از تنوع (Diversity) است.

مثال‌ها شامل موارد زیر هستند:

- پاسخ‌های سبک طبقه‌بندی (Classification-Style Answers)
- استخراج ساختاریافته (Structured Extraction)
- قالب‌بندی قطعی (Deterministic Formatting)
- تبدیل‌های ساده کد (Simple Code Transformations)
- پاسخ‌های مقید دستیار (Constrained Assistant Responses)

---

## ۶. نمونه‌برداری (Sampling)

نمونه‌برداری (Sampling) توکن بعدی را بر اساس توزیع احتمال (Probability Distribution) انتخاب می‌کند.

به‌جای آن‌که همیشه محتمل‌ترین توکن انتخاب شود، سامانه از میان توکن‌های ممکن نمونه‌برداری می‌کند:

$$
\mathrm{next\_token} \sim p(\mathrm{token}|\mathrm{context})
$$

نمونه‌برداری (Sampling) تنوع (Diversity) ایجاد می‌کند. یک پرامپت یکسان می‌تواند در اجراهای مختلف، خروجی‌های متفاوتی تولید کند.

مزایا:

- خروجی‌های متنوع‌تر
- مفید برای نوشتار خلاقانه (Creative Writing)
- می‌تواند از برخی تله‌های رمزگشایی حریصانه (Greedy-Decoding Traps) جلوگیری کند

محدودیت‌ها:

- کمتر قطعی (Less Deterministic)
- ممکن است توکن‌های کم‌کیفیت‌تری تولید کند
- می‌تواند ریسک توهم (Hallucination Risk) را افزایش دهد
- بازتولید دقیق آن سخت‌تر است

نمونه‌برداری (Sampling) معمولاً با پارامترهایی مثل دما (Temperature)، بالا-k (Top-k) و بالا-p (Top-p) کنترل می‌شود.

---

## ۷. دما (Temperature)

دما (Temperature) تیزی یا پخش‌شدگی توزیع احتمال (Probability Distribution) را کنترل می‌کند.

یک تبدیل رایج به صورت زیر است:

$$
\mathrm{adjusted\_logits} = \frac{\mathrm{logits}}{\mathrm{temperature}}
$$

سپس سافت‌مکس (Softmax) اعمال می‌شود.

اثرات:

| دما (Temperature) | رفتار (Behavior) |
|---|---|
| پایین (Low) | قطعی‌تر، متمرکزتر، محافظه‌کارانه‌تر |
| متوسط (Medium) | متعادل |
| بالا (High) | تصادفی‌تر، متنوع‌تر، پرریسک‌تر |

اگر دما (Temperature) به صفر نزدیک شود، رمزگشایی (Decoding) شبیه رمزگشایی حریصانه (Greedy Decoding) می‌شود.

مثال‌ها:

```text
temperature = 0.0  -> قطعی یا تقریباً قطعی (Deterministic or Near-Deterministic)
temperature = 0.2  -> متمرکز (Focused)
temperature = 0.7  -> تولید متعادل (Balanced Generation)
temperature = 1.0  -> توزیع اصلی (Original Distribution)
temperature > 1.0  -> تصادفی‌بودن بیشتر (More Randomness)
```
دما (Temperature) باید بر اساس نیازمندی‌های وظیفه (Task Requirements) انتخاب شود، نه به‌عنوان یک تنظیم همگانی (Universal Setting).

---

## ۸. نمونه‌برداری بالا-k (Top-k Sampling)

نمونه‌برداری بالا-k (Top-k Sampling) توکن‌های نامزد (Candidate Tokens) را به `k` توکن محتمل‌تر محدود می‌کند.

برای مثال، اگر `k = 50` باشد، مدل فقط از میان ۵۰ توکن با بیشترین احتمال نمونه‌برداری می‌کند.

```text
k توکن برتر را نگه دار (Keep Top k Tokens)
احتمال‌ها را دوباره نرمال‌سازی کن (Renormalize Probabilities)
از توکن بعدی نمونه‌برداری کن (Sample Next Token)
```
مزایا:

- توکن‌های بسیار نامحتمل را حذف می‌کند
- خطاهای شدید ناشی از نمونه‌برداری (Extreme Sampling Errors) را کاهش می‌دهد
- درک آن ساده است

محدودیت‌ها:

- `k` ثابت ممکن است در برخی زمینه‌ها بیش‌ازحد محدودکننده باشد
- `k` ثابت ممکن است در زمینه‌های دیگر بیش‌ازحد گسترده باشد
- با توزیع جرم احتمال (Probability Mass Distribution) تطبیق پیدا نمی‌کند

بالا-k (Top-k) مفید است، اما اغلب با کنترل‌های دیگر ترکیب می‌شود.

---

## ۹. نمونه‌برداری بالا-p (Top-p Sampling)

نمونه‌برداری بالا-p (Top-p Sampling) که نمونه‌برداری هسته‌ای (Nucleus Sampling) نیز نامیده می‌شود، کوچک‌ترین مجموعه از توکن‌ها را انتخاب می‌کند که احتمال تجمعی (Cumulative Probability) آن‌ها دست‌کم `p` باشد.

برای مثال:

```text
top_p = 0.9
```
سامانه محتمل‌ترین توکن‌ها را نگه می‌دارد تا مجموع احتمال آن‌ها به ۹۰ درصد برسد، سپس از همان مجموعه نمونه‌برداری می‌کند.

مزایا:

- با میزان اطمینان مدل (Model Confidence) تطبیق پیدا می‌کند
- وقتی توزیع تیز است، گزینه‌های کمتری نگه می‌دارد
- وقتی توزیع پهن است، گزینه‌های بیشتری نگه می‌دارد

محدودیت‌ها:

- اگر توزیع نامطمئن باشد، همچنان ممکن است توکن‌های ضعیفی نمونه‌برداری شوند
- رفتار آن به‌شدت به دما (Temperature) وابسته است
- ممکن است بازتولیدپذیری (Reproducibility) را کاهش دهد

بالا-p (Top-p) معمولاً برای تولید بازپایان (Open-Ended Generation) استفاده می‌شود.

---

## ۱۰. ترکیب پارامترهای رمزگشایی (Combining Decoding Parameters)

دما (Temperature)، بالا-k (Top-k) و بالا-p (Top-p) با هم تعامل دارند.

یک خط لوله معمول نمونه‌برداری (Typical Sampling Pipeline) به صورت زیر است:

```text
لاجیت‌ها (Logits)
-> اعمال دما (Apply Temperature)
-> پالایش با بالا-k یا بالا-p (Filter with Top-k or Top-p)
-> نرمال‌سازی مجدد احتمال‌ها (Renormalize Probabilities)
-> نمونه‌برداری از توکن (Sample Token)
```
تنظیمات نمونه:

| مورد استفاده (Use Case) | دما (Temperature) | بالا-p (Top-p) | نکته (Notes) |
|---|---:|---:|---|
| استخراج قطعی (Deterministic Extraction) | 0.0-0.2 | 1.0 | خروجی‌های پایدار ترجیح داده می‌شوند |
| پرسش‌وپاسخ واقع‌محور (Factual QA) | 0.1-0.4 | 0.8-1.0 | تصادفی‌بودن غیرضروری را کاهش دهید |
| ایده‌پردازی (Brainstorming) | 0.7-1.0 | 0.9-0.95 | تنوع را تشویق کنید |
| نوشتار خلاقانه (Creative Writing) | 0.8-1.2 | 0.9-0.98 | تنوع بیشتر |
| تولید کد (Code Generation) | 0.0-0.4 | 0.8-1.0 | معمولاً دمای پایین‌تر |

این‌ها نقاط شروع (Starting Points) هستند، نه قواعد ثابت. پیکربندی درست باید روی بارکاری هدف (Target Workload) ارزیابی شود.

---

## ۱۱. جست‌وجوی پرتویی (Beam Search)

جست‌وجوی پرتویی (Beam Search) در طول تولید، چندین توالی نامزد (Candidate Sequences) را نگه می‌دارد.

در هر گام، نامزدها را گسترش می‌دهد و بهترین `beam_width` توالی را نگه می‌دارد.

```text
beam_width = 4
۴ تکمیل جزئی برتر را نگه دار (Keep 4 Best Partial Completions)
هر کدام را گسترش بده (Expand Each One)
نامزدها را امتیازدهی کن (Score Candidates)
دوباره ۴ مورد برتر را نگه دار (Keep 4 Best Again)
```
مزایا:

- چندین ادامه با احتمال بالا (High-Probability Continuations) را بررسی می‌کند
- برای برخی وظایف ترجمه (Translation) یا تولید مقید (Constrained Generation) مفید است
- از رمزگشایی حریصانه (Greedy Decoding) نظام‌مندتر است

محدودیت‌ها:

- از رمزگشایی حریصانه (Greedy Decoding) پرهزینه‌تر است
- می‌تواند خروجی‌های کلیشه‌ای یا عمومی (Generic Outputs) تولید کند
- برای گفت‌وگوی بازپایان (Open-Ended Dialogue) همیشه بهتر نیست
- ممکن است متن‌های کوتاه یا با احتمال بالا اما کم‌فایده‌تر را ترجیح دهد

جست‌وجوی پرتویی (Beam Search) در سرویس‌دهی مدل‌های زبانی بزرگ (LLM Serving) به سبک چت (Chat-Style) مدرن، کمتر از نمونه‌برداری (Sampling) یا رمزگشایی حریصانه (Greedy Decoding) رایج است، اما همچنان در برخی تنظیمات خاص تولید توالی (Sequence Generation) مفید باقی می‌ماند.

---

## ۱۲. کنترل‌های تکرار و طول (Repetition and Length Controls)

مدل‌های خودرگرسیو (Autoregressive Models) ممکن است دچار تکرار شوند، به‌ویژه زمانی که تنظیمات رمزگشایی (Decoding Settings) ضعیف باشند یا پرامپت‌ها (Prompts) کیفیت کمی داشته باشند.

کنترل‌های رایج شامل موارد زیر هستند:

| کنترل (Control) | هدف (Purpose) |
|---|---|
| جریمه تکرار (Repetition Penalty) | جریمه‌کردن توکن‌هایی که قبلاً ظاهر شده‌اند |
| جریمه فراوانی (Frequency Penalty) | جریمه‌کردن توکن‌ها بر اساس تعداد تکرار |
| جریمه حضور (Presence Penalty) | جریمه‌کردن توکن‌هایی که دست‌کم یک‌بار ظاهر شده‌اند |
| n-gram بدون تکرار (No-repeat n-gram) | جلوگیری از تکرار بازه‌های توکنی |
| حداکثر توکن‌ها (Maximum Tokens) | توقف پس از یک طول خروجی ثابت |
| حداقل توکن‌ها (Minimum Tokens) | جلوگیری از توقف خیلی زودهنگام |

این کنترل‌ها باید با دقت استفاده شوند. جریمه‌های شدید می‌توانند انسجام (Coherence) را تخریب کنند یا از تکرار لازمِ اصطلاحات در نوشتار فنی، کد یا داده‌های ساختاریافته جلوگیری کنند.

---

## ۱۳. شرایط توقف (Stop Conditions)

تولید (Generation) باید در زمان درست متوقف شود.

شرایط توقف (Stop Conditions) رایج شامل موارد زیر هستند:

- توکن پایان توالی (End-of-Sequence Token)
- حداکثر طول خروجی (Maximum Output Length)
- رشته‌های توقف (Stop Strings)
- مرز فراخوانی ابزار (Tool-Call Boundary)
- تکمیل شیء JSON (JSON Object Completion)
- مرز قالب چت (Chat-Template Boundary)
- لغو خارجی (External Cancellation)
- زمان‌انقضا (Timeout)
- مداخله ایمنی (Safety Intervention)

رشته‌های توقف (Stop Strings) به مدیریت دقیقی نیاز دارند، زیرا ممکن است در مرز میان چند توکن ظاهر شوند.

برای خروجی‌های ساختاریافته (Structured Outputs)، توقف باید با اعتبارسنجی (Validation) هماهنگ شود. ممکن است مدل پس از تولید یک JSON یا کد از نظر نحوی ناقص (Syntactically Incomplete) متوقف شود، مگر این‌که لایه سرویس‌دهی (Serving Layer) نتیجه را بررسی کند.

---

## ۱۴. قالب‌های چت (Chat Templates)

مدل‌های چت (Chat Models) معمولاً با یک قالب گفت‌وگوی مشخص (Specific Conversation Format) آموزش داده می‌شوند.

یک قالب چت (Chat Template)، پیام‌های ساختاریافته (Structured Messages) را به توالی توکنی (Token Sequence) مورد انتظار مدل تبدیل می‌کند.

مثال:

```text
system: شما یک دستیار مفید هستید.
user: توجه (Attention) را توضیح بده.
assistant:
```
قالب سریال‌شده واقعی (Actual Serialized Format) ممکن است شامل توکن‌های ویژه (Special Tokens) زیر باشد:

```text
<|system|>
<|user|>
<|assistant|>
```
استفاده از قالب نادرست (Wrong Template) می‌تواند رفتار مدل را به‌طور قابل‌توجهی تضعیف کند.

خطاهای قالب چت (Chat-Template Errors) می‌توانند باعث موارد زیر شوند:

- سردرگمی نقش‌ها (Role Confusion)
- پیروی ضعیف از دستورالعمل (Poor Instruction Following)
- توقف زودهنگام (Premature Stopping)
- نبود پیشوند دستیار (Missing Assistant Prefix)
- خطاهای قالب‌بندی فراخوانی ابزار (Tool-Call Formatting Errors)
- ناسازگاری در سیاست ایمنی (Safety-Policy Inconsistency)

سامانه استنتاج (Inference System) باید توکنایزر (Tokenizer) و قالب چت (Chat Template) را با هم نسخه‌بندی (Version) کند.

---

## ۱۵. پرامپت‌های سیستمی (System Prompts) و سلسله‌مراتب دستورالعمل (Instruction Hierarchy)

بسیاری از کاربردها (Applications) از پرامپت‌های سیستمی (System Prompts) برای تعریف رفتار دستیار، سبک، محدودیت‌ها و ابزارهای موجود استفاده می‌کنند.

یک سلسله‌مراتب معمول به صورت زیر است:

```text
دستورالعمل‌های سیستمی (System Instructions)
-> دستورالعمل‌های توسعه‌دهنده یا کاربرد (Developer or Application Instructions)
-> دستورالعمل‌های کاربر (User Instructions)
-> زمینه بازیابی‌شده (Retrieved Context)
-> گفت‌وگوی قبلی (Previous Conversation)
-> هدف تولید (Generation Target)
```
مدل این سلسله‌مراتب را به‌طور خودکار اعمال نمی‌کند. مدل الگوها را از آموزش (Training) یاد می‌گیرد و باید با ساخت پرامپت (Prompt Construction)، پالایش (Filtering)، طراحی ابزار (Tool Design) و منطق کاربرد (Application Logic) پشتیبانی شود.

پرامپت‌های سیستمی خوب (Good System Prompts) ویژگی‌های زیر را دارند:

- صریح (Explicit)
- فشرده (Compact)
- پایدار (Stable)
- هم‌راستا با کاربرد (Aligned with the Application)
- آزموده‌شده در برابر ورودی‌های خصمانه و مبهم (Tested Against Adversarial and Ambiguous Inputs)

پرامپت‌های سیستمی بیش‌ازحد طولانی یا مبهم (Overly Long or Vague System Prompts) می‌توانند زمینه (Context) را هدر دهند و قابلیت اطمینان (Reliability) را کاهش دهند.

---

## ۱۶. پنجره‌های زمینه (Context Windows)

پنجره زمینه (Context Window) حداکثر تعداد توکن‌هایی است که مدل می‌تواند در یک درخواست پردازش کند.

زمینه کل (Total Context) شامل موارد زیر است:

```text
پرامپت سیستمی (System Prompt)
+ تاریخچه گفت‌وگو (Conversation History)
+ اسناد بازیابی‌شده (Retrieved Documents)
+ خروجی‌های ابزار (Tool Outputs)
+ پیام کاربر (User Message)
+ پاسخ تولیدشده (Generated Answer)
```
یک پنجره زمینه بلند (Long Context Window) امکان ورودی‌های طولانی‌تر را می‌دهد، اما در عین حال مصرف حافظه (Memory Use) و هزینه پیش‌پرکردن (Prefill Cost) را افزایش می‌دهد.

ملاحظات مهم شامل موارد زیر هستند:

- حداکثر طول توالی پشتیبانی‌شده (Maximum Supported Sequence Length)
- کیفیت مؤثر در بردهای بلند (Effective Quality at Long Range)
- رفتار کدگذاری مکانی (Positional Encoding Behavior)
- ترتیب بازیابی (Retrieval Ordering)
- راهبرد برش (Truncation Strategy)
- خلاصه‌سازی گفت‌وگو (Conversation Summarization)
- هزینه هر درخواست (Cost per Request)
- اثر بر تأخیر (Latency Impact)

زمینه بلند (Long Context) جایگزین انتخاب خوب زمینه (Good Context Selection) نیست.

---

## ۱۷. کش کلید-مقدار (Key-Value Cache)

در طول رمزگشایی (Decoding)، هر لایه ترنسفورمر (Transformer Layer) کلیدها (Keys) و مقدارها (Values)ی توجه (Attention) را برای توکن‌های قبلی ذخیره می‌کند.

به این سازوکار، کش کلید-مقدار (Key-Value Cache) یا کش KV (KV Cache) گفته می‌شود.

بدون کش کلید-مقدار (KV Cache)، هر توکن جدید نیاز داشت که توجه (Attention) روی کل پیشوند (Full Prefix) دوباره محاسبه شود.

با کش کلید-مقدار (KV Cache):

```text
پیش‌پرکردن (Prefill):
  کلیدها و مقدارها را برای توکن‌های پرامپت محاسبه کن

رمزگشایی (Decode):
  کلید و مقدار را فقط برای توکن جدید محاسبه کن
  به کلیدها و مقدارهای کش‌شده قبلی توجه کن
```
کش کلید-مقدار (KV Cache) برای تولید خودرگرسیو کارآمد (Efficient Autoregressive Generation) ضروری است.

---

## ۱۸. حافظه کش KV (KV Cache Memory)
```text
حافظه کش KV (KV Cache Memory) با عوامل زیر مقیاس می‌شود:

- اندازه دسته (Batch Size)
- طول توالی (Sequence Length)
- تعداد لایه‌ها (Number of Layers)
- تعداد سرهای KV (Number of KV Heads)
- بُعد سر (Head Dimension)
- دقت (Precision)
```
این موضوع کش KV (KV Cache) را به یک گلوگاه مهم برای تولید با زمینه بلند (Long-Context Generation) و اندازه‌های دسته بزرگ (Large Batch Sizes) تبدیل می‌کند.

راهبردهای کاهش حافظه کش KV (KV Cache Memory) شامل موارد زیر هستند:

- توجه پرس‌وجوی گروه‌بندی‌شده (Grouped-Query Attention) و توجه چندپرس‌وجویی (Multi-Query Attention)
- کوانتیزاسیون کش KV (Quantization of KV Cache)
- توجه صفحه‌بندی‌شده (PagedAttention) در vLLM
- دسته‌بندی پویا (Dynamic Batching)

---

## ۱۹. رمزگشایی حدسی (Speculative Decoding)

رمزگشایی حدسی (Speculative Decoding) که رمزگشایی کمکی (Assisted Decoding) نیز نامیده می‌شود، استنتاج (Inference) را با استفاده از یک مدل پیش‌نویس (Draft Model) کوچک‌تر و سریع‌تر برای پیش‌بینی چند توکن جلوتر، شتاب می‌دهد.

فرآیند به این صورت است:

1. مدل پیش‌نویس (Draft Model) یک توالی از $N$ توکن تولید می‌کند.
2. مدل اصلی (Main Model) که بزرگ‌تر است، این $N$ توکن را در یک گذر رو‌به‌جلوی موازی (Single Parallel Forward Pass) اعتبارسنجی می‌کند.
3. توکن‌های درست پذیرفته می‌شوند و توکن‌های نادرست رد می‌شوند.
4. مدل اصلی (Main Model) توکن بعدی را تولید می‌کند.

این کار می‌تواند تعداد گذرهای رو‌به‌جلوی ترتیبی (Sequential Forward Passes) موردنیاز را به‌طور معناداری کاهش دهد و گذردهی (Throughput) را افزایش دهد.

مثال:

text
پرامپت (Prompt): The quick brown fox

مدل پیش‌نویس (Draft Model) پیش‌بینی می‌کند: jumps over the
مدل اصلی (Main Model) "jumps"، "over" و "the" را به‌صورت موازی اعتبارسنجی می‌کند
اگر همه درست باشند، مدل اصلی توکن بعدی را تولید می‌کند
اگر "the" نادرست باشد، مدل اصلی بازگشت می‌کند و از "over" ادامه می‌دهد

رمزگشایی حدسی (Speculative Decoding) زمانی بهترین عملکرد را دارد که مدل پیش‌نویس (Draft Model) پیش‌بین خوبی برای توکن‌های بعدی مدل اصلی باشد.

---

## ۲۰. کرنل‌های بهینه‌سازی‌شده (Optimized Kernels)

بهینه‌سازی‌های نرم‌افزاری سطح پایین (Low-Level Software Optimizations) یا کرنل‌ها (Kernels) برای کارایی استنتاج (Inference Performance) حیاتی هستند.

مثال‌ها:

- فلش‌اتنشن (FlashAttention) برای توجه (Attention) سریع‌تر و کم‌حافظه‌تر
- عملگرهای ادغام‌شده (Fused Operators) برای ترکیب چند عملیات در یک کرنل
- روتین‌های بهینه‌شده جبر خطی (Optimized Linear Algebra Routines) مانند `cuBLAS`، `cuDNN` و `xformers`
- کرنل‌های کوانتیزاسیون (Quantization Kernels) مانند `bitsandbytes`
- توجه تنک کارآمد (Efficient Sparse Attention) برای مدل‌های ترکیب خبرگان (Mixture-of-Experts / MoE)

این بهینه‌سازی‌ها می‌توانند بدون تغییر معماری مدل (Model Architecture)، چندین برابر افزایش سرعت ایجاد کنند.

---

## ۲۱. دسته‌بندی پیوسته (Continuous Batching)

دسته‌بندی سنتی (Traditional Batching) درخواست‌ها را در گروه‌هایی با اندازه ثابت پردازش می‌کند. اگر یک درخواست زودتر تمام شود، منابع GPU آن تا پایان کندترین درخواست در آن دسته (Batch) هدر می‌رود.

دسته‌بندی پیوسته (Continuous Batching) که دسته‌بندی پویا (Dynamic Batching) یا دسته‌بندی در سطح تکرار (Iteration-Level Batching) نیز نامیده می‌شود، اجازه می‌دهد به‌محض آزادشدن منابع GPU از درخواست‌های تکمیل‌شده، درخواست‌های جدید به دسته (Batch) اضافه شوند.

مزایا:

- استفاده از GPU را به‌شدت افزایش می‌دهد
- گذردهی (Throughput) را افزایش می‌دهد
- تأخیر متوسط (Average Latency) را کاهش می‌دهد

دسته‌بندی پیوسته (Continuous Batching) یکی از تکنیک‌های اصلی در سامانه‌های سرویس‌دهی مدل زبانی بزرگ با کارایی بالا (High-Performance LLM Serving Systems) مانند `vLLM`، `TensorRT-LLM` و `TGI` است.

---

## ۲۲. موازی‌سازی مدل (Model Parallelism)

برای مدل‌های بسیار بزرگی که در حافظه یک GPU جا نمی‌شوند، موازی‌سازی مدل (Model Parallelism) لایه‌ها یا پارامترهای مدل را میان چندین دستگاه توزیع می‌کند.

انواع موازی‌سازی مدل (Model Parallelism):

- **موازی‌سازی خط لوله‌ای (Pipeline Parallelism):** لایه‌های مختلف مدل روی GPUهای متفاوت قرار می‌گیرند. توکن‌ها به‌صورت ترتیبی از میان GPUها عبور می‌کنند.
- **موازی‌سازی تانسوری (Tensor Parallelism):** لایه‌های منفرد، مانند ماتریس‌های وزن بزرگ، میان چند GPU تقسیم می‌شوند. عملیات درون یک لایه به‌صورت موازی اجرا می‌شود.
- **موازی‌سازی خبره‌ای (Expert Parallelism) برای MoE:** خبرگان مختلف در یک لایه ترکیب خبرگان (Mixture-of-Experts Layer) روی GPUهای مختلف قرار می‌گیرند.

موازی‌سازی مدل (Model Parallelism) سربار ارتباطی (Communication Overhead) ایجاد می‌کند، اما سرویس‌دهی مدل‌هایی با میلیاردها یا تریلیون‌ها پارامتر را ممکن می‌سازد.

---

## ۲۳. زمان‌بندی و داوری درخواست‌ها (Request Scheduling and Arbitration)

در یک محیط چندکاربره (Multi-User Environment)، یک سرور استنتاج (Inference Server) باید درخواست‌ها را به‌شکل کارآمد مدیریت و زمان‌بندی کند.

چالش‌های کلیدی:

- **طول‌های متغیر درخواست (Variable Request Lengths):** طول پرامپت‌ها و خروجی‌های تولیدشده می‌تواند بسیار متفاوت باشد و تخصیص منابع را پیچیده کند.
- **مدیریت کش KV (KV Cache Management):** تخصیص و آزادسازی کارآمد حافظه کش KV (KV Cache Memory) میان درخواست‌های متنوع.
- **اولویت‌بندی (Prioritization):** رسیدگی به درخواست‌هایی با توافق‌نامه‌های سطح خدمت (Service-Level Agreements / SLAs) متفاوت.
- **انصاف (Fairness):** اطمینان از این‌که همه کاربران سرویس معقولی دریافت می‌کنند.
- **گذردهی در برابر تأخیر (Throughput vs. Latency):** ایجاد تعادل میان گذردهی کلی سامانه و تأخیر هر درخواست.

زمان‌بندهای پیشرفته (Advanced Schedulers)، مانند توجه صفحه‌بندی‌شده (PagedAttention) در `vLLM`، این جنبه‌ها را با برخورد با حافظه کش KV (KV Cache Memory) به‌عنوان صفحه‌های مجازی (Virtual Pages)، مشابه مدیریت حافظه در سیستم‌عامل، بهینه می‌کنند.

---

## ۲۴. نکات کلیدی (Key Takeaways)

- **استنتاج (Inference) با آموزش (Training) متفاوت است.** استنتاج برای سرویس‌دهی بهینه می‌شود و میان کیفیت (Quality)، تأخیر (Latency)، گذردهی (Throughput) و هزینه (Cost) تعادل برقرار می‌کند.
- **تولید خودرگرسیو (Autoregressive Generation) یک حلقه است.** تولید توکن‌به‌توکن با کش KV (KV Cache) و رمزگشایی حدسی (Speculative Decoding) بهینه می‌شود.
- **راهبردهای رمزگشایی (Decoding Strategies) خروجی را کنترل می‌کنند.** رمزگشایی حریصانه (Greedy)، نمونه‌برداری (Sampling) با دما (Temperature)، بالا-k (Top-k)، بالا-p (Top-p)، و جست‌وجوی پرتویی (Beam Search) مصالحه‌های متفاوتی ارائه می‌دهند.
- **قالب‌های چت (Chat Templates) حیاتی هستند.** استفاده از قالب درست، پیروی از دستورالعمل (Instruction Following) و رفتار سازگار مدل را تضمین می‌کند.
- **پرامپت‌های سیستمی (System Prompts) رفتار مدل را مدیریت می‌کنند.** نقش‌ها، محدودیت‌ها و ابزارها را به‌وضوح تعریف کنید.
- **مدیریت پنجره زمینه (Context Window Management) بسیار مهم است.** میان طول (Length)، کیفیت (Quality)، حافظه (Memory) و هزینه (Cost) تعادل برقرار کنید.
- **کش KV (KV Cache) برای سرعت ضروری است.** اما یک گلوگاه حافظه (Memory Bottleneck) نیز هست و به بهینه‌سازی‌هایی مانند توجه پرس‌وجوی گروه‌بندی‌شده (GQA)، توجه چندپرس‌وجویی (MQA) و توجه صفحه‌بندی‌شده (PagedAttention) نیاز دارد.
- **کارایی با بهینه‌سازی‌های سطح پایین هدایت می‌شود.** کرنل‌های تخصصی (Specialized Kernels) مانند فلش‌اتنشن (FlashAttention) و دسته‌بندی (Batching) مانند دسته‌بندی پیوسته (Continuous Batching) کلیدی هستند.
- **مدل‌های بزرگ به موازی‌سازی نیاز دارند.** موازی‌سازی مدل (Model Parallelism) از نوع خط لوله‌ای (Pipeline)، تانسوری (Tensor) و خبره‌ای (Expert) سرویس‌دهی مدل‌های عظیم را ممکن می‌کند.
- **زمان‌بندی پیشرفته درخواست‌ها را مدیریت می‌کند.** این کار تخصیص منابع را بهینه می‌کند و تعادل میان گذردهی (Throughput) و تأخیر (Latency) را برقرار می‌سازد.
- **ایمنی (Safety) و قابلیت اطمینان (Reliability) حیاتی هستند.** شرایط توقف (Stop Conditions)، تعدیل محتوا (Content Moderation) و پایش (Monitoring) برای محیط تولید ضروری‌اند.

---
