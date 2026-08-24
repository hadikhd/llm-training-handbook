---
id: pretraining
title: پیش آموزش  
sidebar_label: پیش آموزش
sidebar_position: 6
---
## پیش‌آموزش
<div className="chapter-hero">

![فصل ۵  — پیش آموزش](/img/chapters/pretraining.png)

</div>
[فهرست مطالب](./index.md) |
[قبلی: معماری Transformer](./04-transformer-architecture.md) |
[بعدی: پس‌آموزش و هم‌ترازسازی](./06-post-training-and-alignment.md)

---


## ۱. پیش‌آموزش (Pretraining) به چه معناست

پیش‌آموزش (Pretraining) مرحله‌ای است که در آن یک مدل زبانی، پیش از آن‌که برای وظایف خاص تطبیق داده شود یا با ترجیحات کاربر هم‌راستا (Aligned) گردد، الگوهای عمومی زبانی، دانشی (Factual) و استدلالی (Reasoning) را از متن در مقیاس بزرگ یاد می‌گیرد.

برای LLMهای فقط-رمزگشا (Decoder-only LLMs)، پیش‌آموزش معمولاً خودنظارتی (Self-supervised) است، نه نظارت‌شده (Supervised) به معنای سنتیِ داده‌های برچسب‌خورده. مدل روی نمونه‌های برچسب‌گذاری‌شده‌ی دستی مانند این آموزش نمی‌بیند:
```text
input -> class label
```
در عوض، از دنباله‌های خام توکن (Raw Token Sequences) یاد می‌گیرد و با پیش‌بینی توکن بعدی آموزش داده می‌شود.

```text
The capital of France is -> Paris
```
این هدف خودنظارتیِ ساده، به‌طرز چشمگیری مقیاس‌پذیر (Scalable) است. با داده‌ی کافی، ظرفیت مدل (Model Capacity) مناسب و محاسبات کافی (Compute)، پیش‌بینی توکن بعدی مدل‌هایی تولید می‌کند که نحو (Syntax)، معناشناسی (Semantics)، دانش جهان (World Knowledge)، توانایی ترجمه (Translation Ability)، الگوهای کدنویسی (Coding Patterns) و رفتار استدلالی پایه (Basic Reasoning Behavior) را کسب می‌کنند.

بنابراین، پیش‌آموزش بنیان یک LLM است. پس‌آموزش (Post-training) می‌تواند رفتار مدل را شکل دهد، اما پیش‌آموزش بخش بزرگی از آن‌چه مدل می‌داند و آن‌چه بعداً قادر به یادگیری آن است را تعیین می‌کند.

---


## ۲. هدف پیش‌بینی توکن بعدی (Next-Token Prediction Objective)

برای مدل‌های زبانی فقط-رمزگشا (Decoder-only Language Models)، هدف غالب در پیش‌آموزش، مدل‌سازی زبانی علّی (Causal Language Modeling) است.

برای یک دنباله از توکن‌ها:

$$
x = [x_1, x_2, x_3, \ldots, x_T]
$$

مدل طوری آموزش می‌بیند که هر توکن را از روی توکن‌های پیش از آن پیش‌بینی کند:

$$
P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

هدف آموزشی (Training Target) به‌سادگی همان دنباله است که یک موقعیت جابه‌جا شده است.

```text
input:  [The, cat, sat, on, the]
target: [cat, sat, on, the, mat]
```
به همین دلیل است که ماسک‌گذاری علّی (Causal Masking) ضروری است. در موقعیت `t`، مدل نباید توکن‌های بعد از `t` را ببیند.

بدون ماسک‌گذاری علّی، مدل می‌تواند با توجه مستقیم به توکن‌های آینده تقلب کند؛ در نتیجه loss آموزشی پایین می‌آید، اما مدل به‌عنوان یک مولد خودبازگشتی (Autoregressive Generator) شکست می‌خورد.

---


## ۳. از اسناد (Documents) تا جریان‌های توکن (Token Streams)

داده‌ی آموزشی خام معمولاً به‌صورت اسناد (Documents) آغاز می‌شود: کتاب‌ها، مقاله‌ها، فایل‌های کد، صفحات وب، paperها، مکالمه‌ها یا corpusهای دامنه‌محور (Domain-specific Corpora).

پیش از پیش‌آموزش، این اسناد به جریان‌های توکن (Token Streams) تبدیل می‌شوند:

```text
documents -> cleaning -> filtering -> deduplication -> tokenization -> packed sequences
```
مدل روی اسناد به‌عنوان اشیای با طول متغیر (Variable-length Objects) آموزش نمی‌بیند. سیستم آموزشی، دنباله‌های توکن را ارائه می‌دهد که معمولاً در پنجره‌هایی با طول ثابت (Fixed-length Windows) چیده می‌شوند، مانند:

```text
sequence length = 2048, 4096, 8192, ...
```
یک پرسش کلیدی مهندسی این است که مرزهای سند (Document Boundaries) چگونه مدیریت می‌شوند. برخی pipelineها یک توکن پایان سند (End-of-document Token) درج می‌کنند:

```text
<eos>
```
این کار به مدل می‌آموزد که یک سند کجا تمام می‌شود و سند بعدی از کجا شروع می‌شود. بدون مدیریت دقیق مرزها، ممکن است اسناد نامرتبط به‌طور مصنوعی به هم متصل به نظر برسند.

---


## ۴. بسته‌بندی دنباله (Sequence Packing)

پیش‌آموزش پرهزینه است، بنابراین توکن‌های بلااستفاده اتلاف محسوب می‌شوند.

اگر هر سند کوتاه تا طول کامل context با padding پر شود، ممکن است بخش بزرگی از محاسبات صرف پیش‌بینی توکن‌های padding شود. بسته‌بندی دنباله (Sequence Packing) این مسئله را با الحاق (Concatenating) چندین سند در یک دنباله‌ی آموزشی حل می‌کند.

مثال:

```text
doc1 <eos> doc2 <eos> doc3 <eos>

Then the packed stream is split into fixed-length chunks.
```
این کار بهره‌برداری از سخت‌افزار (Hardware Utilization) را بهبود می‌دهد، اما باید با دقت انجام شود. اگر مرزهای سند نمایش داده نشوند، مدل ممکن است گذارهای غیرطبیعی میان متن‌های نامرتبط را یاد بگیرد.

برای داده‌های چت (Chat)، دستورالعملی (Instruction) یا کد (Code)، مرزها حتی مهم‌تر هستند، زیرا ساختار نمونه (Structure of the Sample) حامل معنا است.

---


## ۵. تابع هزینه (Loss Function): آنتروپی متقاطع (Cross-Entropy)

تابع هزینه‌ی استاندارد در پیش‌آموزش، آنتروپی متقاطع در سطح توکن (Token-level Cross-Entropy) است.

در هر موقعیت، مدل یک توزیع احتمال روی واژگان (Vocabulary) تولید می‌کند:

```text
vocab_size = 32k, 50k, 100k, ...
```
اگر توکن صحیح بعدی `y` باشد، loss برابر است با:

$$
\mathcal{L}_t = -\log P(y_t \mid x_{<t})
$$

loss نهایی روی تعداد زیادی توکن میانگین گرفته می‌شود.

loss کمتر یعنی مدل به توکن‌های صحیح بعدی احتمال بیشتری اختصاص می‌دهد. با این حال، loss پایین به‌طور خودکار به این معنا نیست که مدل مفید (Helpful)، حقیقت‌گو (Truthful)، ایمن (Safe) یا هم‌راستا (Aligned) است. فقط یعنی توزیع آموزشی را به‌خوبی پیش‌بینی می‌کند.

---


# ۶. پرپلکسیتی (Perplexity)

پرپلکسیٹی (Perplexity) یک معیار رایج است که از آنتروپی متقاطع (Cross-Entropy) مشتق می‌شود:


$$
\mathrm{PPL} = \exp(\mathcal{L})
$$

به‌صورت شهودی، پرپلکسیٹی میزان عدم‌قطعیت (Uncertainty) مدل را هنگام پیش‌بینی توکن بعدی اندازه می‌گیرد.

پرپلکسیٹی 20 متناظر با یک loss نماییِ میانگین برابر با 20 است؛ این معیار برای شهود درباره‌ی عدم‌قطعیت پیش‌بینی مفید است، اما نباید به‌صورت تحت‌اللفظی این‌گونه تفسیر شود که مدل به‌طور یکنواخت میان 20 توکن انتخاب می‌کند.

پرپلکسیٹی برای مقایسه‌ی مدل‌ها روی یک tokenizer یکسان و یک corpus ارزیابی یکسان مفید است. اما هنگام مقایسه‌ی مدل‌هایی با tokenizerهای متفاوت، ترکیب داده‌ای متفاوت یا پیش‌پردازش ارزیابی متفاوت، کمتر قابل اتکا است.

---


# ۷. اندازه‌ی بچ (Batch Size)، طول دنباله (Sequence Length) و تعداد توکن در هر به‌روزرسانی (Tokens per Update)

در آموزش LLM، واحد واقعی مقیاس معمولاً توکن است، نه نمونه (Example).

با نادیده‌گرفتن padding و توکن‌های mask‌شده، تعداد اسمی توکن‌هایی که در هر به‌روزرسانی optimizer ارائه می‌شوند برابر است با:


$$
N_{\text{tokens/update}} = B_{\text{global}} \times T
$$

برای مثال:

```text
global_batch_size = 1024
sequence_length = 4096
```
$$
N_{\text{tokens/update}} = 1024 \times 4096 = 4{,}194{,}304
$$

بچ‌های بزرگِ توکنی (Large Token Batches) می‌توانند آموزش را پایدارتر کنند، اما هم‌زمان به حافظه‌ی بیشتر و هماهنگی توزیع‌شده‌ی بیشتری نیاز دارند.

کمیت‌های مهم شامل این موارد‌اند:

- `micro_batch_size`: تعداد دنباله‌هایی که هر دستگاه پیش از انباشت گرادیان (Gradient Accumulation) پردازش می‌کند
- `gradient_accumulation_steps`: تعداد micro-batchهایی که پیش از یک گام optimizer انباشته می‌شوند
- `global_batch_size`: اندازه‌ی کل بچ روی همه‌ی دستگاه‌ها
- `sequence_length`: تعداد توکن‌ها در هر دنباله
- `tokens_per_update`: تعداد کل توکن‌های آموزشی در هر گام optimizer

یک اجرای آموزشیِ عملی (Practical Training Run) اغلب توسط حافظه‌ی GPU، پهنای باند interconnect و راهبرد checkpointing محدود می‌شود.

---


## ۸. بهینه‌ساز (Optimizer): AdamW

AdamW یک optimizer پرکاربرد برای پیش‌آموزش LLM است.

Adam ممان‌های اول (First Moments) و دوم (Second Moments) گرادیان‌ها را برآورد می‌کند:

```text
m_t: running average of gradients
v_t: running average of squared gradients
```
AdamW نسخه‌ای از Adam است که کاهش وزن (Weight Decay) را از به‌روزرسانی گرادیان جدا می‌کند (Decoupling). این کار معمولاً تعمیم‌پذیری (Generalization) را بهتر می‌کند و تنظیم regularization را ساده‌تر می‌سازد.

ابرپارامترهای (Hyperparameters) رایج شامل این موارد‌اند:

```text
learning_rate
betas
weight_decay
epsilon
```
یک پیکربندی معمول ممکن است چنین باشد:

```text
optimizer: AdamW
betas: (0.9, 0.95)
weight_decay: 0.1
epsilon: 1e-8
```
مقادیر دقیق به اندازه‌ی مدل، کیفیت داده، اندازه‌ی batch و مدت آموزش بستگی دارند.

---


## ۹. زمان‌بندی نرخ یادگیری (Learning-Rate Schedule)

نرخ یادگیری (Learning Rate) یکی از مهم‌ترین کنترل‌های آموزشی است.

در پیش‌آموزش LLM معمولاً از این الگو استفاده می‌شود:

```text
warmup -> decay
```
در مرحله‌ی warmup، نرخ یادگیری به‌تدریج از یک مقدار کوچک به مقدار بیشینه (Peak Value) افزایش می‌یابد. این کار ناپایداری اوایل آموزش را کاهش می‌دهد؛ زمانی که وزن‌ها هنوز به‌خوبی کالیبره نشده‌اند.

پس از warmup، نرخ یادگیری کاهش می‌یابد. برنامه‌های زمانی (Schedules) رایج شامل این‌ها هستند:

- cosine decay
- linear decay
- constant with cooldown
- inverse square-root decay

یک الگوی ساده چنین است:

```text
small LR -> peak LR -> slowly decreasing LR
```
نرخ یادگیری بیش‌ازحد بالا می‌تواند باعث واگرایی (Divergence) شود. نرخ یادگیری بیش‌ازحد پایین نیز محاسبات را هدر می‌دهد و ممکن است به کم‌آموزش‌دیدگی مدل (Undertraining) منجر شود.

---


## ۱۰. برش گرادیان (Gradient Clipping)

برش گرادیان (Gradient Clipping) از بی‌ثبات‌شدن آموزش بر اثر گرادیان‌های غیرمعمولاً بزرگ جلوگیری می‌کند.

یک راهبرد رایج، برش بر اساس نرم سراسری (Global Norm Clipping) است:


$$
g \leftarrow
g \cdot
\frac{\tau}{\lVert g\rVert}
\qquad
\text{if } \lVert g\rVert > \tau
$$

مقادیر رایج برای clipping اغلب در حدود زیر هستند:

```text
1.0
```
برش گرادیان جایگزین نرخ یادگیری مناسب، داده‌ی پاک یا معماری پایدار نیست. بلکه یک سازوکار ایمنی (Safety Mechanism) است که کمک می‌کند جهش‌های نادر (Rare Spikes) به اجرای آموزش آسیب نزنند.

---


## ۱۱. آموزش با دقت ترکیبی (Mixed Precision Training)

LLMها به‌ندرت با دقت کامل FP32 آموزش داده می‌شوند، زیرا این کار بیش‌ازحد پرهزینه است.

قالب‌های رایج دقت (Precision Formats) شامل این موارد‌اند:

- FP16
- BF16
- FP32 برای برخی حالت‌های optimizer یا عملیات حساس

BF16 به‌طور گسترده استفاده می‌شود، زیرا همان عرض نما (Exponent Width) را مانند FP32 دارد و بنابراین دامنه‌ی دینامیکی (Dynamic Range) بسیار بزرگ‌تری نسبت به FP16 فراهم می‌کند؛ این ویژگی معمولاً پایداری عددی (Numerical Robustness) را در آموزش در مقیاس بزرگ بهبود می‌دهد.

دقت ترکیبی (Mixed Precision) مصرف حافظه را کاهش می‌دهد و throughput را افزایش می‌دهد، اما پایداری عددی باید با دقت پایش شود.

نشانه‌های رایج مشکلات مربوط به precision شامل این موارد‌اند:

- جهش ناگهانی loss
- NaNها
- exploding gradients
- ناپایداری امتیازهای attention
- خرابی حالت optimizer (Optimizer State Corruption)

---


## ۱۲. پیش‌آموزش توزیع‌شده (Distributed Pretraining)

LLMهای بزرگ معمولاً روی یک GPU واحد قابل آموزش نیستند. آموزش توزیع‌شده، محاسبه و حافظه را میان چندین دستگاه تقسیم می‌کند.

شکل‌های رایج موازی‌سازی (Parallelism) شامل این‌ها هستند:

| Method | Purpose |
|---|---|
| Data Parallelism | Replicate model, split batches |
| Tensor Parallelism | Split large tensors across devices |
| Pipeline Parallelism | Split model layers across devices |
| Sequence Parallelism | Split sequence-related activations |
| ZeRO / FSDP | Shard optimizer states, gradients, and parameters |

موازی‌سازی داده (Data Parallelism) از نظر مفهومی ساده است: هر GPU داده‌ی متفاوتی را پردازش می‌کند و سپس گرادیان‌ها همگام‌سازی (Synchronized) می‌شوند.

موازی‌سازی tensor و pipeline زمانی اهمیت پیدا می‌کنند که خود مدل روی یک دستگاه جا نشود.

روش‌های FSDP و سبک ZeRO با شاردکردن (Sharding) حالت‌های مدل در میان دستگاه‌ها، تکرار حافظه را کاهش می‌دهند.

---


## ۱۳. چک‌پوینت‌گیری (Checkpointing)

چک‌پوینت‌گیری (Checkpointing) ضروری است، زیرا اجرای پیش‌آموزش ممکن است روزها، هفته‌ها یا ماه‌ها طول بکشد.

یک training checkpoint مقاوم (Robust) معمولاً شامل این موارد است:

- وزن‌های مدل (Model Weights)
- حالت optimizer (Optimizer State)
- حالت زمان‌بند نرخ یادگیری (Learning-rate Scheduler State)
- حالت مولد اعداد تصادفی (Random Number Generator State)
- موقعیت data loader
- نسخه یا fingerprint توکنایزر
- پیکربندی آموزش

بدون حالت checkpoint مرتبط، ازسرگیری آموزش (Resuming Training) ممکن است همان مسیر بهینه‌سازی (Optimization Trajectory) یا همان ترتیب داده را بازتولید نکند.

معمولاً دو نوع checkpoint وجود دارد:

| Type | Purpose |
|---|---|
| Training checkpoint | Resume training exactly |
| Release checkpoint | Use model for inference or post-training |

Training checkpointها اغلب بسیار بزرگ‌تر هستند، زیرا حالت‌های optimizer را نیز شامل می‌شوند.

---


## ۱۴. پایش آموزش (Monitoring Training)

پیش‌آموزش باید به‌طور پیوسته پایش شود.

سیگنال‌های مهم شامل این موارد‌اند:

- training loss
- validation loss
- learning rate
- gradient norm
- throughput
- GPU memory usage
- token/sec
- نسبت‌های ترکیب داده (Data Mixture Proportions)
- loss به‌تفکیک دامنه‌ی داده (Loss by Data Domain)
- شناسایی NaN یا Inf
- سلامت checkpoint

داشتن یک منحنی loss صاف کافی نیست. یک مدل می‌تواند به‌صورت پایدار آموزش ببیند، در حالی که از داده‌های کم‌کیفیت، تکراری (Duplicated)، آلوده (Contaminated) یا بدترکیب (Badly Mixed) یاد می‌گیرد.

پایش در سطح دامنه (Domain-level Monitoring) به‌ویژه مهم است. اگر کد (Code)، ریاضی (Math)، فارسی (Persian)، انگلیسی (English) یا متن‌های بلند (Long-form Text) در یک اجرای واحد ترکیب شده باشند، هر دامنه باید جداگانه ردیابی شود.

---


## ۱۵. شکست‌های رایج در پیش‌آموزش (Common Pretraining Failures)


## مشکلات داده (Data Problems)

داده‌ی ضعیف معمولاً به مدل ضعیف منجر می‌شود، حتی اگر معماری و optimizer درست باشند.

مسائل رایج شامل این موارد‌اند:

- تکرار بیش‌ازحد (Excessive Duplication)
- آلودگی بنچمارک (Benchmark Contamination)
- نرمال‌سازی شکسته‌ی Unicode
- شناسایی نادرست زبان (Bad Language Identification)
- OCR کم‌کیفیت
- داده‌ی سمی (Toxic) یا سرشار از اسپم (Spam-heavy)
- کد بدشکل (Malformed Code)
- خرابی مرزهای سند (Document Boundary Corruption)


## مشکلات بهینه‌سازی (Optimization Problems)

آموزش ممکن است به دلیل بهینه‌سازی ناپایدار شکست بخورد.

نشانه‌های رایج شامل این‌ها هستند:

- واگرایی loss
- جهش‌های مکرر loss
- انفجار گرادیان (Gradient Explosions)
- NaNها
- حساسیت به راه‌اندازی مجدد (Sensitivity to Restart)
- mixed precision ناپایدار


## مشکلات مقیاس‌گذاری (Scaling Problems)

سیستم‌های توزیع‌شده حالت‌های شکست مخصوص به خود را ایجاد می‌کنند.

نمونه‌ها شامل این موارد‌اند:

- ارتباط interconnect کند
- ساخت batch ناکارآمد
- گلوگاه‌های checkpoint
- استفاده‌ی پایین از GPU
- گرسنگی data loader
- seedهای تصادفی ناسازگار میان workerها

---


## ۱۶. چک‌لیست عملی پیش‌آموزش (Practical Pretraining Checklist)

پیش از شروع یک اجرای جدی پیش‌آموزش، این موارد را بررسی کنید:

- tokenizer نهایی و versioned شده است.
- توکن‌های ویژه (Special Tokens) تثبیت شده‌اند.
- پاک‌سازی داده و حذف تکرار (Deduplication) کامل شده است.
- مرزهای سند به‌درستی نمایش داده می‌شوند.
- Sequence packing آزموده شده است.
- Loss masking برای هر نوع توکنی که باید یا نباید در هدف آموزشی مشارکت کند، درست است.
- Causal masking درست است.
- مجموعه‌های آموزش و اعتبارسنجی از هم جدا هستند.
- بررسی‌های benchmark contamination انجام شده‌اند.
- تنظیمات optimizer و scheduler ثبت شده‌اند.
- Mixed precision در مقیاس کوچک آزموده شده است.
- ازسرگیری checkpoint پیش از اجرای کامل آزموده شده است.
- Throughput اندازه‌گیری شده است.
- داشبوردهای پایش آماده‌اند.
- یک آزمون overfitting در مقیاس کوچک موفق بوده است.
- یک اجرای آزمایشی کوتاه (Short Pilot Run) یک منحنی loss سالم تولید می‌کند.

---
#
# ۱۷. نکات کلیدی (Key Takeaways)

پیش‌آموزش مرحله‌ای است که در آن یک LLM قابلیت‌های گسترده‌ی خود را از جریان‌های خام توکن (Raw Token Streams) یاد می‌گیرد.

هدف غالب، پیش‌بینی توکن بعدی (Next-token Prediction) همراه با ماسک‌گذاری علّی (Causal Masking) است.

چالش اصلی مهندسی فقط طراحی مدل نیست، بلکه هماهنگ‌سازی کیفیت داده، پایداری tokenizer، sequence packing، بهینه‌سازی، سیستم‌های توزیع‌شده و پایش است.

یک اجرای موفق پیش‌آموزش باید بازتولیدپذیر (Reproducible)، مشاهده‌پذیر (Observable)، پایدار (Stable) و آگاه از داده (Data-aware) باشد.

پس‌آموزش (Post-training) می‌تواند رفتار مدل را بهبود دهد، اما نمی‌تواند به‌طور کامل یک بنیان ضعیف در پیش‌آموزش را جبران کند.

---

