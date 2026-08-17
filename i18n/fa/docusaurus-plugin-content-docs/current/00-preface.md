---
id: preface
title: پیشگفتار
sidebar_label: پیشگفتار
sidebar_position: 1
---
<div className="chapter-hero">

![Preface](/static/img/chapters/preface.png)

</div>

[فهرست مطالب](./00-preface.md) |
[محتوا](./index.md) |
[بعدی: مقدمه](./01-introduction.md)

---

# پیش‌گفتار

## ۱. هدف این هندبوک

این هندبوک یک مقدمه‌ی عملی و فنی بر Large Language Model (مدل‌های زبانی بزرگ) است.

این متن برای خوانندگانی نوشته شده است که می‌خواهند درک کنند LLMهای مدرن چگونه ساخته می‌شوند، آموزش می‌بینند، ارزیابی می‌شوند، بهینه‌سازی می‌شوند و استقرار پیدا می‌کنند. هدف فقط توصیف مفاهیم در سطح بالا نیست، بلکه پیوند دادن آن‌ها به تصمیم‌های مهندسی‌ای است که سامانه‌های واقعی مدل را شکل می‌دهند.

مدل‌های زبانی بزرگ اغلب طوری مطرح می‌شوند که گویی یک شیء واحد هستند. در عمل، یک LLM system (سامانه‌ی LLM) یک پشته از لایه‌ها و مؤلفه‌ها است:
```text
data
-> tokenization
-> model architecture
-> pretraining
-> scaling decisions
-> post-training
-> evaluation
-> serving infrastructure
-> inference optimization
-> product integration
-> monitoring

```
درک این پشته ضروری است. رفتار یک مدل فقط توسط Parameter Count (تعداد پارامترها) تعیین نمی‌شود. این رفتار تحت تأثیر Data Quality (کیفیت داده)، Tokenizer Design (طراحی توکنایزر)، Architecture Choices (انتخاب‌های معماری)، Optimization Details (جزئیات بهینه‌سازی)، Alignment Methods (روش‌های هم‌راستاسازی)، Decoding Settings (تنظیمات رمزگشایی/تولید)، Retrieval Systems (سامانه‌های بازیابی)، Deployment Constraints (قیود استقرار) و Evaluation Practices (شیوه‌های ارزیابی) شکل می‌گیرد.

این هندبوک همین مسیر کامل را دنبال می‌کند.

---

## ۲. این هندبوک برای چه کسانی است

این هندبوک برای گروه‌های زیر طراحی شده است:
- مهندس یادگیری ماشین (Machine Learning Engineer)
- مهندس یادگیری عمیق (Deep Learning Engineer)
- پژوهشگر هوش مصنوعی (AI Researcher)
- دانشمند داده (Data Scientist) که در حال ورود به حوزهٔ مدل‌های زبانی بزرگ (LLMs) است
- مهندس نرم‌افزار (Software Engineer) که در حال ساخت کاربردهای مدل‌های زبانی بزرگ (LLM Applications) است
- رهبر فنی (Technical Lead) که سامانه‌های مدل‌های زبانی بزرگ (LLM Systems) را ارزیابی می‌کند
- دانشجویانی که با مفاهیم پایهٔ یادگیری ماشین (Machine Learning — ML) آشنا هستند و به دنبال یک مسیر ساختاریافته برای ورود به مهندسی مدل‌های زبانی بزرگ (LLM Engineering) هستند

این مطالب فرض می‌کند که خواننده با مفاهیم پایه‌ی Machine Learning (یادگیری ماشین) آشنا است، از جمله:

- داده‌های آموزشی (Training Data)
- داده‌های اعتبارسنجی (Validation Data)
- تابع زیان (Loss Function)
- گرادیان نزولی (Gradient Descent)
- شبکهٔ عصبی (Neural Network)
- تعبیه / بردار تعبیه (Embedding)
- توزیع احتمال (Probability Distribution)
- بیش‌برازش (Overfitting)
- معیار ارزیابی (Evaluation Metric)

داشتن تجربه‌ی قبلی با Transformer (ترنسفورمر) مفید است، اما الزامی نیست. معماری Transformer در یک فصل مستقل توضیح داده می‌شود.

---

## ۳. این هندبوک چه موضوعاتی را پوشش می‌دهد

این هندبوک به‌صورت یک روند پیوسته از مبانی تا استقرار سازمان‌دهی شده است.

فصل‌ها عبارت‌اند از:

| فصل | موضوع |
|---|---|
| 00 | پیش‌گفتار |
| 01 | مقدمه |
| 02 | داده |
| 03 | توکن‌سازی |
| 04 | معماری ترنسفورمر |
| 05 | پیش‌آموزش |
| 06 | قوانین مقیاس‌پذیری |
| 07 | پس‌آموزش |
| 08 | ارزیابی |
| 09 | سامانه‌ها |
| 10 | استنتاج و رمزگشایی |
| 11 | چشم‌انداز |
| 12 | واژه‌نامه |

ساختار این کتاب از Lifecycle (چرخه‌عمر) یک LLM پیروی می‌کند:

```text
define the problem
-> collect and prepare data
-> convert text into tokens
-> train a Transformer
-> scale under compute constraints
-> align the model for use
-> evaluate behavior
-> build the serving system
-> optimize inference
-> plan for future directions
-> standardize terminology

```
هر فصل روی ایده‌هایی تمرکز دارد که در عمل بیشترین اهمیت را دارند.

---

## ۴. این هندبوک چه کاری را انجام نمی‌دهد

این هندبوک یک Research Survey (مرور جامع پژوهشی) کامل نیست.

این کتاب تلاش نمی‌کند همه‌ی مقاله‌ها، همه‌ی خانواده‌های مدل، همه‌ی Benchmark (بنچمارک)ها یا همه‌ی Training Trick (ترفندهای آموزشی) را پوشش دهد. حوزه‌ی LLM آن‌قدر سریع تغییر می‌کند که چنین رویکردی خیلی زود کارایی خود را از دست می‌دهد.

در عوض، این هندبوک روی Durable Concepts (مفاهیم ماندگار) تمرکز دارد:

- چرا کیفیت داده (Data Quality) اهمیت دارد
- چرا توکن‌سازی (Tokenization) بر رفتار مدل اثر می‌گذارد
- چرا ترنسفورمرها (Transformers) به‌خوبی مقیاس‌پذیر می‌شوند
- چرا پیش‌آموزش (Pretraining) معمولاً بر پایهٔ پیش‌بینی توکن بعدی (Next-Token Prediction) است
- چرا قوانین مقیاس‌پذیری (Scaling Laws) تخصیص محاسبات (Compute) را هدایت می‌کنند
- چرا پس‌آموزش (Post-Training) قابلیت استفاده از مدل را تغییر می‌دهد
- چرا ارزیابی (Evaluation) باید به‌عنوان یک سامانهٔ مهندسی (Engineering System) دیده شود
- چرا آموزش توزیع‌شده (Distributed Training) و سامانه‌های استنتاج (Inference Systems) اهمیت دارند
- چرا تنظیمات رمزگشایی (Decoding Settings) بر کیفیت خروجی اثر می‌گذارند
- چرا سامانه‌های آیندهٔ مدل‌های زبانی بزرگ (Large Language Models — LLMs) یکپارچه‌تر، چندوجهی (Multimodal) و ابزارمحور (Tool-Using) خواهند بود


تأکید اصلی بر درک مکانیزم‌های زیربنایی است تا بتوان روش‌های جدید را به‌صورت انتقادی ارزیابی کرد.

---

## ۵. راهبرد مطالعه

ترتیب پیشنهادی مطالعه، ترتیبی و گام‌به‌گام است.

ابتدا از مقدمه شروع کنید، سپس به‌ترتیب به سراغ داده، توکن‌سازی، معماری، آموزش، مقیاس‌پذیری، پس‌آموزش، ارزیابی، سامانه‌ها، استنتاج، چشم‌انداز و در نهایت واژه‌نامه بروید.

یک مسیر مناسب برای مطالعه به این صورت است:

```text
01 Introduction
-> 02 Data
-> 03 Tokenization
-> 04 Transformer Architecture
-> 05 Pretraining
-> 06 Scaling Laws
-> 07 Post-training
-> 08 Evaluation
-> 09 Systems
-> 10 Inference and Decoding
-> 11 Outlook
-> 12 Glossary

```
خوانندگانی که از قبل تجربه دارند می‌توانند از این هندبوک به‌عنوان Reference (مرجع) نیز استفاده کنند. برای مثال:

- اگر در حال ساخت یک مجموعه‌داده (Dataset) هستید، از فصل ۲ (Chapter 2) شروع کنید.
- اگر در حال مطالعهٔ درونیات مدل (Model Internals) هستید، روی فصل ۳ (Chapter 3) و فصل ۴ (Chapter 4) تمرکز کنید.
- اگر به آموزش مدل (Training) علاقه‌مند هستید، فصل ۵ (Chapter 5)، فصل ۶ (Chapter 6) و فصل ۹ (Chapter 9) را بخوانید.
- اگر روی یک مدل گفت‌وگومحور (Chat Model) کار می‌کنید، فصل ۷ (Chapter 7) را بخوانید.
- اگر در حال استقرار (Deployment) یک مدل هستید، فصل ۸ (Chapter 8)، فصل ۹ (Chapter 9) و فصل ۱۰ (Chapter 10) را بخوانید.
- اگر در حال طراحی یک محصول مبتنی بر مدل زبانی بزرگ (LLM Product) هستید، فصل ۷ (Chapter 7)، فصل ۸ (Chapter 8)، فصل ۱۰ (Chapter 10) و فصل ۱۱ (Chapter 11) را بخوانید.
- اگر برای شفاف‌سازی اصطلاحات (Terminology) به مرجع نیاز دارید، از واژه‌نامهٔ پایانی (Glossary) استفاده کنید.

---

## ۶. تم مرکزی

تم مرکزی این هندبوک این است:

```text
LLMs are not only models.
They are engineered systems.
```
یک محصول LLM که در محیط واقعی استقرار یافته باشد ممکن است شامل موارد زیر باشد:

- مدل پایه (Base Model)
- توکنایزر (Tokenizer)
- قالب گفت‌وگو (Chat Template)
- سازندهٔ پرامپت (Prompt Builder)
- خط لولهٔ بازیابی (Retrieval Pipeline)
- بازمرتب‌ساز (Reranker)
- فشرده‌ساز زمینه (Context Compressor)
- لایهٔ فراخوانی ابزار (Tool-Calling Layer)
- فیلتر ایمنی (Safety Filter)
- پیکربندی رمزگشایی (Decoding Configuration)
- سرور استنتاج (Inference Server)
- کش (Cache)
- مجموعهٔ ارزیابی (Evaluation Suite)
- داشبورد پایش (Monitoring Dashboard)
- سامانهٔ لاگ‌گیری و بازخورد (Logging and Feedback System)

بهبود چنین سامانه‌ای به‌ندرت تنها به یک تکنیک وابسته است. این کار نیازمند شناسایی Bottleneck (گلوگاه) فعلی است.

برای مثال:

| مسئله | ناحیه‌ی محتمل برای بررسی |
|---|---|
| خطاهای factual (واقعیت‌محور) | داده، بازیابی، ارزیابی، رمزگشایی |
| قالب‌بندی ضعیف | پس‌آموزش، قالب‌ها، پرامپت‌ها |
| Latency (تأخیر) بالا | استنتاج، batching، KV cache، quantization |
| دانش دامنه‌ای ضعیف | داده، RAG، fine-tuning |
| رفتار چندزبانه‌ی ضعیف | ترکیب داده، توکنایزر، ارزیابی |
| پاسخ‌های ناپایدار | رمزگشایی، ارزیابی، پس‌آموزش |
| هزینه‌ی بالای سرویس‌دهی | اندازه مدل، routing، caching، compression |

یک مهندس عملی این پرسش را مطرح می‌کند:

```text
Which layer is failing, and what evidence shows that?
```
---

## ۷. سطح ریاضی

این هندبوک از Mathematics (ریاضیات) در جایی استفاده می‌کند که به شفاف‌سازی مفهوم کمک کند، اما از Formalism (صورت‌بندی صوری) غیرضروری پرهیز می‌کند.

شما با ایده‌هایی از این جنس مواجه خواهید شد:

- توزیع احتمال (Probability Distribution) روی توکن‌ها
- زیان آنتروپی متقاطع (Cross-Entropy Loss)
- تعبیه (Embedding)
- امتیاز توجه (Attention Score)
- سافت‌مکس (Softmax)
- بهینه‌سازی (Optimization)
- رابطهٔ مقیاس‌پذیری (Scaling Relationship)
- پیچیدگی حافظه (Memory Complexity) و پیچیدگی محاسباتی (Compute Complexity)


هدف این نیست که هر نتیجه از اصول اولیه اثبات شود. هدف این است که درک ریاضی کافی برای استدلال درباره‌ی رفتار مدل و Engineering Trade-off (موازنه‌های مهندسی) ایجاد شود.

هر زمان که یک فرمول ظاهر می‌شود، باید به یک پرسش عملی پاسخ دهد:

```text
What is being optimized?
What is being measured?
What grows with model size?
What changes when sequence length increases?
What cost does this design introduce?
```
---

## ۸. ملاحظات عملی

LLM Engineering (مهندسی LLM) پر از Caveat (ملاحظه و ظرافت اجرایی) است.

یک تکنیک ممکن است در یک شرایط بسیار خوب کار کند و در شرایطی دیگر شکست بخورد. یک Decoding Setting (تنظیم رمزگشایی)، Prompt Strategy (راهبرد پرامپت)، Retrieval Method (روش بازیابی) یا Fine-tuning Recipe (نسخه‌ی تنظیم دقیق) نباید به‌عنوان یک راه‌حل universally correct (همه‌جا درست) در نظر گرفته شود.

برخی ملاحظات مهم عبارت‌اند از:

- بهبود بنچمارک (Benchmark) همیشه به محیط تولید (Production) منتقل نمی‌شود.
- مدل‌های بزرگ‌تر همیشه برای یک وظیفه (Task) خاص بهتر نیستند.
- زمینه (Context) بیشتر، تضمین‌کنندهٔ استفادهٔ بهتر از اطلاعات نیست.
- دادهٔ مصنوعی (Synthetic Data) می‌تواند فرایند آموزش را بهبود دهد یا خطاها را تقویت کند.
- کوانتیزاسیون (Quantization) می‌تواند هزینه را کاهش دهد، اما ممکن است رفتار مدل را تغییر دهد.
- تولید افزوده با بازیابی (Retrieval-Augmented Generation — RAG) می‌تواند توهم (Hallucination) را کاهش دهد، اما در عین حال ممکن است خطای بازیابی (Retrieval Error) ایجاد کند.
- مدل زبانی بزرگ به‌عنوان داور (LLM-as-a-Judge) می‌تواند به فرایند ارزیابی کمک کند، اما ممکن است دچار سوگیری (Bias) باشد.
- ریزتنظیم (Fine-Tuning) می‌تواند سبک یا عملکرد مدل را در یک وظیفه بهبود دهد، اما ممکن است به توانایی‌های عمومی مدل آسیب بزند.
- عامل‌ها (Agents) به قیود ابزار (Tool Constraints)، مدیریت حالت (State Management) و ارزیابی (Evaluation) نیاز دارند.
- کیفیت محیط تولید (Production Environment) فقط با آزمون آفلاین (Offline Testing) به‌دست نمی‌آید و به پایش (Monitoring) نیاز دارد.


به همین دلیل، این هندبوک بر Measurement (اندازه‌گیری) و Iteration (تکرار و بهبود مستمر) تأکید می‌کند.

یک Workflow (گردش‌کار) قابل اتکا به این صورت است:

```text
build a baseline
-> evaluate it
-> identify failure modes
-> change one component
-> evaluate again
-> deploy cautiously
-> monitor real behavior
```
---

## ۹. اصطلاحات

تیم‌ها و مقاله‌های مختلف ممکن است برای ایده‌های مشابه از اصطلاحات متفاوتی استفاده کنند.

برای مثال:

- پس‌آموزش (Post-Training) ممکن است شامل تنظیم دقیق نظارت‌شده (Supervised Fine-Tuning)، بهینه‌سازی ترجیحی (Preference Optimization) و یادگیری تقویتی (Reinforcement Learning) باشد.
- هم‌راستاسازی (Alignment) ممکن است به مفیدبودن (Helpfulness)، بی‌ضرربودن (Harmlessness)، پیروی از دستور (Instruction Following) یا کنترل گسترده‌تر رفتار مدل اشاره داشته باشد.
- طول زمینه (Context Length) ممکن است به حد معماری (Architectural Limit)، حد قابل استفاده (Usable Limit) یا پیکربندی سرویس‌دهی (Serving Configuration) اشاره داشته باشد.
- ارزیابی (Evaluation) ممکن است به امتیاز بنچمارک (Benchmark Score)، بازبینی انسانی (Human Review)، آزمون رگرسیون (Regression Test) یا پایش در محیط تولید (Production Monitoring) اشاره داشته باشد.
- تولید افزوده با بازیابی (Retrieval-Augmented Generation — RAG) ممکن است به یک پرامپت ساده مبتنی بر جست‌وجوی برداری (Vector-Search Prompt) یا یک سامانهٔ کامل بازیابی و مبتنی‌سازی بر شواهد (Retrieval and Grounding System) اشاره داشته باشد.

این هندبوک تلاش می‌کند اصطلاحات را به‌شکل سازگار استفاده کند و آن‌ها را در متن توضیح دهد.

هرجا که Terminology (اصطلاح‌شناسی) در سطح حوزه متغیر باشد، معنای عملی بر نام‌گذاری سخت‌گیرانه اولویت دارد.

---

## ۱۰. چگونه از بلوک‌های شبه‌کدی استفاده کنیم

این هندبوک از Code-like Block (بلوک‌های شبیه کد) برای نمایش فشرده‌ی نمودارهای مفهومی استفاده می‌کند.

مثال:

```text
documents
-> cleaning
-> deduplication
-> tokenization
-> training batches
```
این بلوک‌ها همیشه Code (کد) قابل اجرا نیستند. هدف آن‌ها نمایش Flow (جریان)، Dependency (وابستگی) یا Structure (ساختار) است.

از Table (جدول) برای مقایسه‌ی Trade-off (موازنه‌ها) استفاده می‌شود. از List (فهرست) برای صریح‌کردن Checklist (چک‌لیست)های مهندسی استفاده می‌شود.

---

## ۱۱. پرسش‌های راهنما

در حین مطالعه، پرسش‌های زیر را در ذهن داشته باشید:

- مدل بر روی چه داده‌هایی (Data) آموزش دیده است؟
- متن چگونه به توکن (Token) تبدیل شده است؟
- چه معماری‌ای (Architecture) دنباله را پردازش می‌کند؟
- چه هدف آموزشی (Training Objective)ای بهینه‌سازی شده است؟
- چه بودجهٔ محاسباتی (Compute Budget)ای مدل را شکل داده است؟
- مدل پایه (Base Model) چگونه برای استفادهٔ موردنظر تطبیق داده شده است؟
- کیفیت مدل چگونه ارزیابی (Evaluation) می‌شود؟
- مدل چگونه به‌صورت کارآمد سرویس‌دهی (Serving) می‌شود؟
- تأخیر (Latency) و هزینه (Cost) چگونه کنترل می‌شوند؟
- وقتی مدل دچار خطا می‌شود، چه اتفاقی می‌افتد؟
- سامانه پس از استقرار (Deployment) چگونه پایش (Monitoring) می‌شود؟
- کدام اصطلاحات نیاز به شفاف‌سازی در واژه‌نامه (Glossary) دارند؟

این پرسش‌ها از حفظ کردن تکنیک‌های منفرد مفیدتر هستند.

---

## ۱۲. نکته پایانی

Large Language Model (مدل‌های زبانی بزرگ) قدرتمند هستند، زیرا مقیاس، داده، معماری، بهینه‌سازی و انضباط مهندسی را با هم ترکیب می‌کنند.

در عین حال، آن‌ها محدودیت هم دارند. آن‌ها ممکن است Hallucinate (دچار توهم شوند)، Overfit (بیش‌برازش داشته باشند)، Context (زمینه) را نادرست تفسیر کنند، در Tool Use (استفاده از ابزار) شکست بخورند، استدلال ناسازگار تولید کنند، یا با تغییرات کوچک در Prompt (پرامپت) یا Decoding Setting (تنظیمات رمزگشایی) رفتار متفاوتی نشان دهند.

هدف این هندبوک آن است که این نقاط قوت و محدودیت‌ها را قابل فهم کند.

یک LLM Engineer (مهندس LLM) خوب رفتار مدل را جادویی تلقی نمی‌کند. او داده را بررسی می‌کند، خروجی‌ها را اندازه می‌گیرد، محدودیت‌های سامانه را می‌فهمد و خط لوله را بر پایه‌ی Evidence (شواهد) بهبود می‌دهد.

این همان ذهنیتی است که این هندبوک بر پایه‌ی آن ساخته شده است.

---

[فهرست مطالب](./00-preface.md) |
[فهرست](./index.md) |
[بعدی: مقدمه](./01-introduction.md)
