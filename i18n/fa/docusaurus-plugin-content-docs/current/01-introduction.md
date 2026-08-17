---
id: مقدمه 
title: مقدمه 
sidebar_label: مقدمه 
sidebar_position: 1
---
<div className="chapter-hero">

![Introduction](/img/chapters/introduction.png)

</div>

## ۱.۱ مدل زبانی چیست؟ (What Is a Language Model?)

[قبلی: پیشگفتار (Preface)](./01-preface.md) |
[فهرست (Contents)](./index.md) |
[بعدی: توکن‌سازی (Tokenization)](./02-data.md)


یک مدل زبانی (Language Model) به دنباله‌هایی از توکن‌ها (Tokens) احتمال اختصاص می‌دهد.

فرض کنید یک دنبالهٔ توکن (Token Sequence) به‌صورت زیر باشد:

$$
x_{1:T} = (x_1, x_2, \ldots, x_T)
$$

یک مدل زبانی (Language Model) احتمال مشترک (Joint Probability) را به‌صورت زیر برآورد می‌کند:

$$
P(x_{1:T})
$$

با استفاده از قانون زنجیره‌ای احتمال (Probability Chain Rule)، این توزیع مشترک (Joint Distribution) را می‌توان به‌صورت زیر تجزیه کرد:

$$
P(x_{1:T})
=
\prod_{t=1}^{T}
P(x_t \mid x_{1:t-1})
$$

که در آن:

$$
x_{1:t-1} = (x_1, x_2, \ldots, x_{t-1})
$$

بنابراین مدل یاد می‌گیرد هر توکن (Token) را از روی توکن‌هایی که پیش از آن آمده‌اند پیش‌بینی کند.

برای مثال:

```text
Context: The capital of France is
Target:  Paris

The model computes a probability distribution over its vocabulary:

Paris       0.81
Lyon        0.04
France      0.03
London      0.01
...
```
در هنگام تولید (Generation)، یک توکن (Token) از این توزیع انتخاب شده و به زمینه (Context) افزوده می‌شود. این فرایند تا زمانی تکرار می‌شود که مدل یک توکن توقف (Stopping Token) تولید کند یا به یک محدودیت تولید (Generation Limit) برسد.

---

## ۱.۲ متن، توکن‌ها و توزیع‌های احتمال (Text, Tokens, and Probability Distributions)

مدل‌های زبانی (Language Models) مستقیماً روی واژه‌ها یا نویسه‌ها عمل نمی‌کنند. یک توکنایزر (Tokenizer) متن را به یک دنباله از شناسه‌های گسستهٔ توکن (Discrete Token Identifiers) تبدیل می‌کند.

برای مثال:

```text
Text:
Transformers process sequences efficiently.

Possible tokens:
["Transform", "ers", " process", " sequences", " efficiently", "."]

Possible token IDs:
[41762, 388, 1920, 16374, 17226, 13]
```
بخش‌بندی دقیق به توکنایزر (Tokenizer) و واژگان (Vocabulary) آن بستگی دارد.

اگر واژگان (Vocabulary) شامل $V$ توکن باشد، مدل در هر موقعیت از دنباله $V$ لاجیت (Logits) تولید می‌کند:

$$
z_t \in \mathbb{R}^{V}
$$

یک تابع سافت‌مکس (Softmax Function) این لاجیت‌ها (Logits) را به احتمال تبدیل می‌کند:

$$
P(x_t = i \mid x_{<t})
=
\frac{\exp(z_{t,i})}
{\sum_{j=1}^{V}\exp(z_{t,j})}
$$

بردار حاصل یک توزیع احتمال دسته‌ای (Categorical Probability Distribution) را روی توکن بعدی (Next Token) تعریف می‌کند.

---

## ۱.۳ مدل‌سازی زبانی خودرگرسیو (Autoregressive Language Modeling)

بیشتر مدل‌های زبانی بزرگ (Large Language Models - LLMs) برای تولید متن از یک هدف خودرگرسیو (Autoregressive Objective) استفاده می‌کنند.

خودرگرسیو (Autoregressive) به این معناست که مدل یک دنباله را هر بار یک عنصر تولید می‌کند و هر پیش‌بینی جدید را به عناصر در دسترس قبلی شرطی می‌کند.

با داشتن:

```text
Large language models

فرایند تولید (Generation Process) ممکن است به این شکل باشد:

text
Step 1: Large language models are
Step 2: Large language models are trained
Step 3: Large language models are trained on
Step 4: Large language models are trained on large
Step 5: Large language models are trained on large datasets
```
با این حال، در زمان آموزش (Training)، معمولاً می‌توان پیش‌بینی‌های تمام موقعیت‌ها را به‌صورت موازی محاسبه کرد، زیرا کل دنبالهٔ هدف (Target Sequence) از قبل در دسترس است.

یک ماسک توجه علّی (Causal Attention Mask) تضمین می‌کند که موقعیت $t$ نتواند به موقعیت‌های آینده دسترسی داشته باشد:

$$
x_t \text{ may attend only to } x_1, \ldots, x_t
$$

این موضوع هدف خودرگرسیو (Autoregressive Objective) را حفظ می‌کند و در عین حال آموزش موازی کارآمد (Efficient Parallel Training) را ممکن می‌سازد.

---

## ۱.۴ از مدل‌های N-Gram تا مدل‌های زبانی عصبی (From N-Grams to Neural Language Models)

### ۱.۴.۱ مدل‌های N-Gram (N-Gram Models)

مدل‌های زبانی آماری سنتی (Traditional Statistical Language Models) احتمال توکن بعدی (Next-Token Probability) را فقط با استفاده از تعداد ثابتی از توکن‌های قبلی تقریب می‌زنند.

یک مدل سه‌گرمی (Trigram Model) تقریب می‌زند:

$$
P(x_t \mid x_{<t})
\approx
P(x_t \mid x_{t-2}, x_{t-1})
$$

احتمال‌های آن را می‌توان با استفاده از شمارش‌های پیکره (Corpus Counts) برآورد کرد:

$$
P(x_t \mid x_{t-2}, x_{t-1})
=
\frac{
C(x_{t-2}, x_{t-1}, x_t)
}{
C(x_{t-2}, x_{t-1})
}
$$

مدل‌های N-Gram از نظر مفهومی ساده هستند، اما محدودیت‌های مهمی دارند:

* آن‌ها از یک زمینهٔ کوتاه و ثابت (Short, Fixed Context) استفاده می‌کنند.
* جدول‌های شمارش (Count Tables) آن‌ها با اندازهٔ واژگان (Vocabulary Size) به‌سرعت رشد می‌کند.
* بسیاری از دنباله‌های معتبر هرگز در پیکرهٔ آموزشی (Training Corpus) رخ نمی‌دهند.
* آن‌ها به‌خوبی به ترکیب‌های دیده‌نشده تعمیم نمی‌یابند.
* واژه‌های مشابه به‌طور خودکار قدرت آماری (Statistical Strength) را با هم به اشتراک نمی‌گذارند.

روش‌های هموارسازی (Smoothing Methods) مشکلات احتمال صفر (Zero-Probability Problems) را کاهش می‌دهند، اما محدودیت‌های بنیادی را از بین نمی‌برند.

### ۱.۴.۲ مدل‌های زبانی عصبی (Neural Language Models)

مدل‌های زبانی عصبی (Neural Language Models) جدول‌های شمارش تنک (Sparse Count Tables) را با نمایش‌های پیوستهٔ یادگرفته‌شده (Learned Continuous Representations) جایگزین می‌کنند.

توکن‌ها به امبدینگ‌ها (Embeddings) نگاشت می‌شوند:

$$
e_t = E[x_t]
$$

که در آن $E$ یک ماتریس امبدینگ (Embedding Matrix) است.

توکن‌هایی با کارکرد یا معنای مشابه می‌توانند نمایش‌های مرتبطی ایجاد کنند. بنابراین مدل می‌تواند در میان عبارت‌هایی که دقیقاً با همان شکل در زمان آموزش ظاهر نشده‌اند تعمیم دهد.

مدل‌های زبانی عصبی اولیه (Early Neural Language Models) از شبکه‌های پیش‌خور (Feed-forward Networks) استفاده می‌کردند. بعدها شبکه‌های عصبی بازگشتی (Recurrent Neural Networks)، حافظهٔ بلند-کوتاه‌مدت (LSTMs) و واحدهای بازگشتی دروازه‌دار (GRUs) با حفظ یک حالت پنهان (Hidden State)، مدل‌سازی دنباله (Sequence Modeling) را بهبود دادند.

با این حال، مدل‌های بازگشتی (Recurrent Models) توکن‌ها را به‌صورت ترتیبی پردازش می‌کنند و این موضوع آموزش موازی در مقیاس بزرگ (Large-Scale Parallel Training) را دشوار می‌سازد. همچنین آن‌ها ممکن است در حفظ اطلاعات در فاصله‌های بسیار طولانی دچار مشکل شوند.

---

## ۱.۵ چرا ترنسفورمرها مدل‌سازی زبانی را تغییر دادند؟ (Why Transformers Changed Language Modeling)

ترنسفورمر (Transformer) بازگشت (Recurrence) را با پردازش دنباله مبتنی بر توجه (Attention-Based Sequence Processing) جایگزین کرد.

عملیات مرکزی آن خودتوجهی (Self-Attention) است:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V
$$

که در آن:

* $Q$ شامل بردارهای پرسش (Query Vectors) است.
* $K$ شامل بردارهای کلید (Key Vectors) است.
* $V$ شامل بردارهای مقدار (Value Vectors) است.
* $d_k$ بُعد کلید (Key Dimension) است.
* $M$ یک ماسک اختیاری (Optional Mask) مانند ماسک علّی (Causal Mask) است.

ترنسفورمرها به معماری غالب مدل‌های زبانی بزرگ (Dominant LLM Architecture) تبدیل شدند، زیرا موارد زیر را فراهم می‌کنند:

* آموزش موازی کارآمد در سراسر موقعیت‌های دنباله (Efficient Parallel Training Across Sequence Positions)
* تعامل انعطاف‌پذیر بین توکن‌های دور از هم (Flexible Interaction Between Distant Tokens)
* رفتار مقیاس‌پذیری قوی (Strong Scaling Behavior)
* سازگاری با ضرب ماتریسی بسیار بهینه‌شده (Highly Optimized Matrix Multiplication)
* یک معماری ماژولار (Modular Architecture) که می‌تواند روی شتاب‌دهنده‌های متعدد توزیع شود

ترنسفورمر استاندارد (Standard Transformer) همچنان محدودیت‌هایی دارد. به‌ویژه، خودتوجهی معمولی (Ordinary Self-Attention) نسبت به طول دنباله دارای پیچیدگی درجه‌دوم (Quadratic Complexity) است:

$$
O(T^2)
$$

این موضوع با افزایش طول زمینه (Context Length) هم بر محاسبه و هم بر حافظه اثر می‌گذارد.

---

## ۱.۶ چه چیزی یک مدل زبانی را «بزرگ» می‌کند؟ (What Makes a Language Model "Large"?)

هیچ آستانهٔ جهانی برای تعداد پارامترها (Parameter Threshold) وجود ندارد که یک مدل زبانی بزرگ (LLM) را تعریف کند.

این اصطلاح معمولاً به یک مدل زبانی عصبی (Neural Language Model) اشاره دارد که ترکیبی از موارد زیر را دارد:

* تعداد پارامتر زیاد (A Large Parameter Count)
* پیکرهٔ آموزشی بزرگ و متنوع (A Large and Diverse Training Corpus)
* محاسبات آموزشی قابل‌توجه (Significant Training Compute)
* قابلیت‌های گسترده در چندین وظیفه (Broad Capabilities Across Multiple Tasks)

تعداد پارامتر به‌تنهایی کافی نیست. یک مدل بزرگ که با دادهٔ ناکافی یا کم‌کیفیت آموزش دیده باشد، ممکن است از یک مدل کوچک‌تر اما خوب‌آموزش‌دیده ضعیف‌تر عمل کند.

کیفیت مدل به تعامل میان موارد زیر بستگی دارد:

$$
\text{Capability}
=
f(
\text{architecture},
\text{parameters},
\text{data},
\text{compute},
\text{optimization},
\text{post-training}
)
$$

دو مدل با تعداد پارامتر یکسان ممکن است به‌دلیل تفاوت در کیفیت داده، توکن‌سازی (Tokenization)، طول زمینه (Context Length)، بهینه‌سازی (Optimization) یا پس‌آموزش (Post-training) رفتارهای بسیار متفاوتی داشته باشند.

---

## ۱.۷ پارامترها، فعال‌سازی‌ها و ابرپارامترها (Parameters, Activations, and Hyperparameters)

این مفاهیم نباید با یکدیگر اشتباه گرفته شوند.

### پارامترها (Parameters)

پارامترها (Parameters) مقادیر یادگرفته‌شده هستند، از جمله:

* ماتریس‌های امبدینگ (Embedding Matrices)
* ماتریس‌های فرافکنی توجه (Attention Projection Matrices)
* وزن‌های شبکهٔ پیش‌خور (Feed-forward Network Weights)
* پارامترهای نرمال‌سازی (Normalization Parameters)
* وزن‌های فرافکنی خروجی (Output Projection Weights)

این‌ها در طول آموزش (Training) به‌روزرسانی می‌شوند.

### فعال‌سازی‌ها (Activations)

فعال‌سازی‌ها (Activations) مقادیر میانی هستند که در طول یک گذر رو‌به‌جلو (Forward Pass) تولید می‌شوند.

مصرف حافظهٔ آن‌ها به‌شدت به موارد زیر بستگی دارد:

* اندازهٔ بچ (Batch Size)
* طول دنباله (Sequence Length)
* بُعد پنهان (Hidden Dimension)
* تعداد لایه‌ها (Number of Layers)
* پیاده‌سازی توجه (Attention Implementation)
* راهبرد بازمحاسبهٔ فعال‌سازی (Activation Checkpointing Strategy)

### ابرپارامترها (Hyperparameters)

ابرپارامترها (Hyperparameters) توسط تیم آموزش انتخاب می‌شوند، نه اینکه مستقیماً یاد گرفته شوند.

نمونه‌ها عبارت‌اند از:

* نرخ یادگیری (Learning Rate)
* اندازهٔ بچ (Batch Size)
* تعداد لایه‌ها (Number of Layers)
* بُعد پنهان (Hidden Dimension)
* تعداد سرهای توجه (Number of Attention Heads)
* اندازهٔ واژگان (Vocabulary Size)
* طول دنباله (Sequence Length)
* افت وزن (Weight Decay)
* مدت گرم‌سازی (Warmup Duration)

انتخاب ابرپارامترها (Hyperparameter Choices) بر پایداری، کارایی و کیفیت نهایی مدل اثر می‌گذارد.

---

## ۱.۸ چرخهٔ توسعهٔ مدل زبانی بزرگ (The LLM Development Lifecycle)

یک خط لولهٔ عملی توسعهٔ مدل زبانی بزرگ (Practical LLM Development Pipeline) شامل چندین مرحله است.

### مرحلهٔ ۱: گردآوری داده (Stage 1: Data Acquisition)

منابع ممکن شامل موارد زیر هستند:

* اسناد وب (Web Documents)
* کتاب‌ها (Books)
* متون علمی (Scientific Literature)
* کد منبع (Source Code)
* مواد مرجع (Reference Material)
* گفتگوها (Conversations)
* مجموعه‌داده‌های خصوصی دارای مجوز (Licensed Private Datasets)
* نمونه‌های مصنوعی (Synthetic Examples)

### مرحلهٔ ۲: پردازش داده (Stage 2: Data Processing)

عملیات معمول شامل موارد زیر است:

* استخراج متن (Text Extraction)
* شناسایی زبان (Language Identification)
* پالایش کیفیت (Quality Filtering)
* پالایش ایمنی (Safety Filtering)
* حذف تکرار دقیق (Exact Deduplication)
* حذف موارد تقریباً تکراری (Near-Duplicate Removal)
* رسیدگی به اطلاعات هویتی شخصی (Personally Identifiable Information Handling)
* رفع آلودگی بنچمارک (Benchmark Decontamination)
* ترکیب مجموعه‌داده (Dataset Mixing)

### مرحلهٔ ۳: آموزش توکنایزر (Stage 3: Tokenizer Training)

توکنایزر (Tokenizer) موارد زیر را تعریف می‌کند:

* واحدهای واژگانی (Vocabulary Units)
* نگاشت توکن به شناسه (Token-to-ID Mapping)
* توکن‌های ویژه (Special Tokens)
* قواعد نرمال‌سازی متن (Text Normalization Rules)
* رفتار بخش‌بندی چندزبانه (Multilingual Segmentation Behavior)

### مرحلهٔ ۴: طراحی مدل (Stage 4: Model Design)

معماری (Architecture) موارد زیر را تعیین می‌کند:

* تعداد پارامترها (Parameter Count)
* تعداد لایه‌ها (Number of Layers)
* اندازهٔ پنهان (Hidden Size)
* پیکربندی توجه (Attention Configuration)
* ابعاد شبکهٔ پیش‌خور (Feed-forward Dimensions)
* نمایش موقعیتی (Positional Representation)
* طول زمینه (Context Length)
* توابع نرمال‌سازی و فعال‌سازی (Normalization and Activation Functions)

### مرحلهٔ ۵: پیش‌آموزش (Stage 5: Pretraining)

مدل از طریق اهدافی مانند پیش‌بینی توکن بعدی (Next-Token Prediction) ساختار آماری عمومی را یاد می‌گیرد.

### مرحلهٔ ۶: پس‌آموزش (Stage 6: Post-training)

مدل پیش‌آموزش‌دیده (Pretrained Model) با استفاده از تکنیک‌هایی مانند موارد زیر سازگار می‌شود:

* تنظیم دقیق نظارت‌شده (Supervised Fine-tuning)
* تنظیم دستورالعمل (Instruction Tuning)
* بهینه‌سازی ترجیح (Preference Optimization)
* یادگیری تقویتی (Reinforcement Learning)
* تنظیم ایمنی (Safety Tuning)
* آموزش استفاده از ابزار (Tool-use Training)

### مرحلهٔ ۷: ارزیابی (Stage 7: Evaluation)

ارزیابی حوزه‌هایی مانند موارد زیر را پوشش می‌دهد:

* کیفیت مدل‌سازی زبانی (Language Modeling Quality)
* دانش (Knowledge)
* استدلال (Reasoning)
* کدنویسی (Coding)
* عملکرد چندزبانه (Multilingual Performance)
* پیروی از دستورالعمل (Instruction Following)
* ایمنی (Safety)
* رفتار در زمینهٔ بلند (Long-context Behavior)
* عملکرد ویژهٔ دامنه (Domain-specific Performance)

### مرحلهٔ ۸: استنتاج و سرویس‌دهی (Stage 8: Inference and Serving)

استقرار (Deployment) نیازمندی‌های اضافی زیر را وارد می‌کند:

* تأخیر کم (Low Latency)
* گذردهی بالا (High Throughput)
* کارایی حافظه (Memory Efficiency)
* زمان‌بندی درخواست (Request Scheduling)
* کوانتیزه‌سازی (Quantization)
* مدیریت کش KV (KV-cache Management)
* پایش (Monitoring)
* کنترل هزینه (Cost Control)

---

## ۱.۹ پیش‌آموزش و پس‌آموزش (Pretraining and Post-training)

### پیش‌آموزش (Pretraining)

پیش‌آموزش (Pretraining) معمولاً یک هدف گستردهٔ مدل‌سازی زبانی (Broad Language-Modeling Objective) را روی یک پیکرهٔ بزرگ (Large Corpus) بهینه می‌کند.

هدف آن یادگیری موارد زیر است:

* الگوهای زبانی (Linguistic Patterns)
* ارتباط‌های واقعی (Factual Associations)
* ساختارهای رایج استدلال (Common Reasoning Structures)
* دانش دامنه (Domain Knowledge)
* ویژگی‌های بازنمایی (Representational Features)
* توانایی عمومی ادامه‌دادن متن (General Continuation Ability)

یک مدل پیش‌آموزش‌دیده (Pretrained Model) معمولاً مدل پایه (Base Model) نامیده می‌شود.

### پس‌آموزش (Post-training)

پس‌آموزش (Post-training) رفتار مدل را پس از پیش‌آموزش تغییر می‌دهد.

هدف آن ممکن است شامل موارد زیر باشد:

* بهبود پیروی از دستورالعمل (Improving Instruction Following)
* آموزش قالب‌های گفتگو (Teaching Conversation Formats)
* افزایش سودمندی پاسخ (Increasing Response Usefulness)
* اعمال محدودیت‌های رفتاری (Applying Behavioral Constraints)
* بهبود استفاده از ابزار (Improving Tool Use)
* سازگار شدن با یک دامنهٔ خاص (Adapting to a Specific Domain)

یک مدل پس‌آموزش‌دیده (Post-trained Model) لزوماً دانش واقعی بسیار بیشتری نسبت به مدل پایهٔ خود ندارد. بخش زیادی از این بهبود ممکن است از استخراج بهتر (Better Elicitation) و انتخاب پاسخ بهتر (Better Response Selection) ناشی شود.

---

## ۱.۱۰ آموزش، استنتاج و سرویس‌دهی (Training, Inference, and Serving)

### آموزش (Training)

آموزش (Training) شامل موارد زیر است:

۱. یک گذر رو‌به‌جلو (Forward Pass)
۲. محاسبهٔ زیان (Loss Calculation)
۳. پس‌انتشار (Backpropagation)
۴. همگام‌سازی گرادیان (Gradient Synchronization) در صورت توزیع‌شده بودن
۵. به‌روزرسانی‌های بهینه‌ساز (Optimizer Updates)

آموزش، موارد زیر را ذخیره یا مدیریت می‌کند:

* پارامترهای مدل (Model Parameters)
* فعال‌سازی‌ها (Activations)
* گرادیان‌ها (Gradients)
* وضعیت‌های بهینه‌ساز (Optimizer States)
* بافرهای موقت (Temporary Buffers)

### استنتاج (Inference)

استنتاج (Inference) خروجی‌های مدل را بدون به‌روزرسانی وزن بر پایهٔ گرادیان (Gradient-Based Weight Updates) محاسبه می‌کند.

برای تولید خودرگرسیو (Autoregressive Generation)، استنتاج معمولاً دو فاز دارد:

* **پیش‌پرکنی (Prefill):** پردازش همهٔ توکن‌های پرامپت (Prompt Tokens)، معمولاً به‌صورت موازی.
* **رمزگشایی (Decode):** تولید توکن‌های بیشتر، هر بار یکی.

### سرویس‌دهی (Serving)

سرویس‌دهی (Serving) سامانهٔ تولیدی پیرامون استنتاج است. این سامانه شامل موارد زیر است:

* پذیرش درخواست (Request Admission)
* بچ‌بندی پویا یا پیوسته (Dynamic or Continuous Batching)
* زمان‌بندی (Scheduling)
* تخصیص کش KV (KV-cache Allocation)
* اجرای موازی مدل (Model Parallel Execution)
* پاسخ‌دهی جریانی (Streaming Responses)
* محدودسازی نرخ (Rate Limiting)
* مدیریت خطا (Failure Handling)
* مشاهده‌پذیری (Observability)

ممکن است یک مدل در یک بنچمارک مجزا سریع باشد، اما اگر سامانهٔ سرویس‌دهی (Serving System) بچ‌بندی یا حافظه را بد مدیریت کند، در محیط تولید (Production) ناکارآمد باشد.

---

## ۱.۱۱ مدل‌های بنیادین، پایه، دستورالعملی و چت (Foundation, Base, Instruction, and Chat Models)

### مدل بنیادین (Foundation Model)

یک مدل گسترده که می‌تواند از چندین کاربرد پایین‌دستی (Downstream Applications) پشتیبانی کند. این اصطلاح ممکن است هم مدل‌های پایه (Base Models) و هم مدل‌های سازگارشده (Adapted Models) را دربر بگیرد.

### مدل پایه (Base Model)

مدلی که عمدتاً با یک هدف عمومی پیش‌آموزش (General Pretraining Objective) آموزش داده شده است.

معمولاً بهتر است آن را یک مدل ادامه‌دهندهٔ متن (Continuation Model) در نظر بگیریم تا یک دستیار گفتگومحور (Conversational Assistant).

### مدل تنظیم‌شده با دستورالعمل (Instruction-Tuned Model)

مدلی که روی نمونه‌های دستورالعمل-پاسخ (Instruction-Response Examples) تنظیم دقیق شده است.

این مدل یاد می‌گیرد درخواست‌ها را تفسیر کند و پاسخ‌های وظیفه‌محور (Task-Oriented Responses) تولید کند.

### مدل چت (Chat Model)

یک مدل تنظیم‌شده با دستورالعمل (Instruction-Tuned Model) که برای تعامل چندنوبتی (Multi-turn Interaction) با استفاده از یک قالب گفتگوی تعریف‌شده (Defined Conversation Template) آموزش دیده است.

این قالب ممکن است شامل نقش‌های ویژه‌ای مانند موارد زیر باشد:

```text
system
user
assistant
tool
```
استفاده از قالب چت نادرست (Wrong Chat Template) حتی زمانی که وزن‌های مدل درست هستند می‌تواند کیفیت را به‌طور قابل‌توجهی کاهش دهد.

---

## ۱.۱۲ قابلیت‌های نوظهور و عمومی (Emergent and General Capabilities)

مدل‌های زبانی بزرگ (LLMs) می‌توانند وظایفی را انجام دهند که به‌صورت ماژول‌های جداگانه نمایش داده نشده‌اند:

* خلاصه‌سازی (Summarization)
* ترجمه (Translation)
* دسته‌بندی (Classification)
* پاسخ به پرسش (Question Answering)
* تولید کد (Code Generation)
* استخراج اطلاعات (Information Extraction)
* یادگیری درون‌زمینه (In-Context Learning)
* تولید خروجی ساخت‌یافته (Structured Output Generation)

این رفتارها از بازنمایی‌های اشتراکی (Shared Representations) و هدف پیش‌بینی توکن بعدی (Next-Token Prediction Objective) پدید می‌آیند.

با این حال، کلیت ظاهری وظایف (Apparent Task Generality) نباید به‌عنوان قابلیت اطمینان همگانی (Universal Reliability) تفسیر شود. عملکرد بسته به موارد زیر تغییر می‌کند:

* قالب پرامپت (Prompt Format)
* زبان (Language)
* دامنه (Domain)
* طول زمینه (Context Length)
* دقت موردنیاز (Required Precision)
* در دسترس بودن الگوهای آموزشی مرتبط (Availability of Relevant Training Patterns)
* روش ارزیابی (Evaluation Method)

---

## ۱.۱۳ یادگیری درون‌زمینه (In-Context Learning)

یادگیری درون‌زمینه (In-Context Learning) زمانی رخ می‌دهد که یک مدل رفتار خود را بر اساس مثال‌هایی که در پرامپت (Prompt) قرار داده شده‌اند سازگار می‌کند، بدون اینکه پارامترهایش را به‌روزرسانی کند.

### بدون‌نمونه (Zero-shot)

```text
Classify the sentiment as positive or negative:

The documentation was clear and useful.
```
### تک‌نمونه (One-shot)

```text
Text: The service was terrible.
Label: negative

Text: The documentation was clear and useful.
Label: positive
```
### چندنمونه (Few-shot)

چندین نمایش (Demonstrations) پیش از مثال هدف (Target Example) ارائه می‌شوند.

یادگیری درون‌زمینه (In-Context Learning) مفید است، اما به موارد زیر حساس است:

* انتخاب مثال (Example Selection)
* ترتیب مثال‌ها (Example Order)
* صورت‌بندی برچسب‌ها (Label Wording)
* قالب‌بندی پرامپت (Prompt Formatting)
* طول زمینه (Context Length)
* ناهماهنگی توزیع (Distribution Mismatch)

این روش نباید با تنظیم دقیق (Fine-tuning) اشتباه گرفته شود، زیرا هیچ به‌روزرسانی پارامتری (Parameter Update) رخ نمی‌دهد.

---

## ۱.۱۴ محدودیت‌های اصلی (Core Limitations)

### توهم (Hallucination)

یک مدل ممکن است ادعاهایی باورپذیر اما بدون پشتوانه تولید کند، زیرا هدف آن ادامه‌های محتمل (Likely Continuations) را پاداش می‌دهد، نه حقیقت تضمین‌شده (Guaranteed Truth).

### زمینهٔ محدود (Finite Context)

مدل فقط می‌تواند مستقیماً به اطلاعاتی شرطی شود که درون پنجرهٔ زمینه (Context Window) آن موجود است.

### کهنگی دانش (Knowledge Staleness)

پارامترها (Parameters) نمایانگر اطلاعاتی هستند که در طول آموزش یاد گرفته شده‌اند. آن‌ها وقتی جهان تغییر می‌کند به‌صورت خودکار به‌روزرسانی نمی‌شوند.

### حساسیت به پرامپت (Prompt Sensitivity)

تغییرات کوچک در واژه‌بندی یا ساختار ممکن است بر کیفیت خروجی اثر بگذارد.

### کالیبراسیون ضعیف (Weak Calibration)

سبک زبانی مطمئن (Confident Linguistic Style) لزوماً نشان‌دهندهٔ احتمال بالای درستی نیست.

### سوگیری و شکاف‌های پوشش (Bias and Coverage Gaps)

مدل‌ها سوگیری‌ها و عدم‌توازن‌ها را از داده، حاشیه‌نویسی‌ها (Annotations)، معیارهای ارزیابی (Evaluation Criteria) و رویه‌های پس‌آموزش (Post-training Procedures) به ارث می‌برند.

### هزینهٔ محاسباتی (Computational Cost)

آموزش و سرویس‌دهی مدل‌های بزرگ به محاسبات، حافظه، پهنای باند ارتباطی و مهندسی عملیاتی قابل‌توجهی نیاز دارد.

### تفسیرپذیری محدود (Limited Interpretability)

ارائهٔ توضیح‌های علّی کامل (Complete Causal Explanations) برای بسیاری از خروجی‌های مدل همچنان دشوار است.

---

## ۱.۱۵ مدل‌های زبانی بزرگ و تولید افزوده با بازیابی (LLMs and Retrieval-Augmented Generation)

تولید افزوده با بازیابی (Retrieval-Augmented Generation) یا RAG، یک مدل زبانی (Language Model) را با یک سامانهٔ بازیابی خارجی (External Retrieval System) ترکیب می‌کند.

یک خط لولهٔ ساده‌شده (Simplified Pipeline) به این صورت است:

```text
User query
   |
   v
Query processing
   |
   v
Retriever
   |
   v
Relevant documents
   |
   v
Prompt construction
   |
   v
Language model
   |
   v
Grounded response
```
RAG می‌تواند موارد زیر را بهبود دهد:

* دسترسی به اطلاعات جاری (Access to Current Information)
* دقت ویژهٔ دامنه (Domain-specific Accuracy)
* انتساب منبع (Source Attribution)
* ممیزی‌پذیری (Auditability)
* پوشش دانشی (Knowledge Coverage)

RAG به‌طور خودکار توهم (Hallucinations) را حذف نمی‌کند. این روش ممکن است به دلایل زیر شکست بخورد:

* بازیابی ضعیف (Poor Retrieval)
* شواهد ناقص (Missing Evidence)
* زمینهٔ نامرتبط (Irrelevant Context)
* برش زمینه (Context Truncation)
* ساخت ضعیف پرامپت (Weak Prompt Construction)
* ناتوانی در پیروی از شواهد (Failure to Follow the Evidence)
* نگاشت نادرست ارجاع‌ها (Incorrect Citation Mapping)

بنابراین بازیاب (Retriever) و مولد (Generator) باید هم به‌صورت جداگانه و هم به‌صورت انتها‌به‌انتها (End to End) ارزیابی شوند.

---

## ۱.۱۶ مثال عملی: یک هدف کمینه برای مدل زبانی علّی (Practical Example: A Minimal Causal LM Objective)

شبه‌کد (Pseudocode) زیر آموزش توکن بعدی (Next-Token Training) را نشان می‌دهد:

```python
input_ids = batch[:, :-1]
target_ids = batch[:, 1:]

logits = model(input_ids)

loss = cross_entropy(
    logits.reshape(-1, vocabulary_size),
    target_ids.reshape(-1),
)

loss.backward()
optimizer.step()
optimizer.zero_grad()
```
دنباله‌های ورودی (Input Sequences) و هدف (Target Sequences) به اندازهٔ یک توکن جابه‌جا می‌شوند:

```text
Input:   [BOS, Large, language, models, are]
Target:  [Large, language, models, are, useful]
```
در کد آموزش تولیدی (Production Training Code)، ملاحظات اضافی شامل موارد زیر است:

* ماسک‌های پدینگ (Padding Masks)
* بسته‌بندی دنباله (Sequence Packing)
* همگام‌سازی توزیع‌شده (Distributed Synchronization)
* دقت ترکیبی (Mixed Precision)
* انباشت گرادیان (Gradient Accumulation)
* برش گرادیان (Gradient Clipping)
* زمان‌بندی نرخ یادگیری (Learning-rate Scheduling)
* چک‌پوینت‌گیری (Checkpointing)
* بازیابی از خطا (Fault Recovery)

---

## ۱.۱۷ یادداشت‌های مهندسی عملی (Practical Engineering Notes)

* طراحی توکنایزر (Tokenizer Design) را بخشی از مدل در نظر بگیرید، نه یک جزئیات پیش‌پردازش قابل‌تعویض (Replaceable Preprocessing Detail).
* هنگام اندازه‌گیری حجم آموزش (Training Volume)، توکن‌ها را ردیابی کنید نه فقط اسناد یا واژه‌ها.
* ارزیابی مدل پایه (Base-model Evaluation) را از ارزیابی پیروی از دستورالعمل (Instruction-following Evaluation) جدا کنید.
* در زمان استنتاج (Inference)، عملکرد پیش‌پرکنی (Prefill) و رمزگشایی (Decode) را به‌صورت مستقل ارزیابی کنید.
* پیش از افزایش اندازهٔ مدل (Model Size)، کیفیت داده و تکرار داده (Duplication) را اندازه‌گیری کنید.
* پیش از عیب‌یابی کیفیت مدل (Model Quality)، قالب‌های چت (Chat Templates) و توکن‌های ویژه (Special Tokens) را اعتبارسنجی کنید.
* نسخهٔ دقیق مجموعه‌داده (Dataset)، کد، توکنایزر (Tokenizer) و چک‌پوینت (Checkpoint) را ثبت کنید.
* در سراسر آموزش (Training)، فقط پس از تکمیل کار ارزیابی نگه‌داشته‌شده (Held-out Evaluations) انجام ندهید.
* محدودیت‌های مدل را از خطاهای بازیابی (Retrieval)، پرامپت‌نویسی (Prompting) و سرویس‌دهی (Serving Failures) متمایز کنید.

---

## ۱.۱۸ خطاهای رایج (Common Pitfalls)

### خطای ۱: برابر دانستن تعداد پارامتر با کیفیت (Equating Parameter Count with Quality)

تعداد پارامتر بیشتر (Larger Parameter Count) تضمین نمی‌کند که مدل بهتری داشته باشید. دادهٔ آموزشی (Training Data)، بودجهٔ توکن (Token Budget)، معماری (Architecture) و کیفیت پس‌آموزش (Post-training Quality) اهمیت دارند.

### خطای ۲: در نظر گرفتن مدل زبانی بزرگ به‌عنوان پایگاه داده (Treating an LLM as a Database)

یک مدل زبانی بزرگ (LLM) یک مولد احتمالاتی (Probabilistic Generator) است. این مدل ذخیره‌سازی تراکنشی (Transactional Storage)، بازیابی تضمین‌شده (Guaranteed Lookup) یا منشأشناسی خودکار (Automatic Provenance) ارائه نمی‌دهد.

### خطای ۳: برابر دانستن خروجی روان با خروجی درست (Treating Fluent Output as Correct Output)

روانی (Fluency) و درستی واقعی (Factual Correctness) دو ویژگی متفاوت هستند.

### خطای ۴: مقایسهٔ مدل‌ها با قالب‌های پرامپت متفاوت (Comparing Models with Different Prompt Formats)

ممکن است یک مدل زمانی ضعیف به نظر برسد که با یک قالب چت نادرست (Incorrect Chat Template) یا راهبرد پرامپت‌نویسی نامناسب (Unsuitable Prompting Strategy) ارزیابی شده باشد.

### خطای ۵: استفاده از پرپلکسیتی به‌عنوان تنها معیار (Using Perplexity as the Only Metric)

پرپلکسیتی (Perplexity) پیش‌بینی توکن بعدی (Next-Token Prediction) را اندازه می‌گیرد، اما استدلال (Reasoning)، پیروی از دستورالعمل (Instruction Following)، ایمنی (Safety) یا استناد واقعی (Factual Grounding) را به‌طور کامل اندازه‌گیری نمی‌کند.

### خطای ۶: نادیده گرفتن آلودگی داده (Ignoring Data Contamination)

هم‌پوشانی با بنچمارک (Benchmark Overlap) ممکن است عملکرد اندازه‌گیری‌شده را بدون بهبود قابلیت عمومی (General Capability) افزایش دهد.

### خطای ۷: نادیده گرفتن محدودیت‌های سامانه‌ای (Ignoring Systems Constraints)

یک معماری از نظر نظری مناسب ممکن است در عمل نامناسب باشد اگر نتوان آن را روی سخت‌افزار موجود (Available Hardware) به‌شکل کارآمد آموزش داد یا سرویس‌دهی کرد.

---

## ۱.۱۹ جمع‌بندی (Summary)

* یک مدل زبانی (Language Model) احتمال‌ها را روی دنباله‌های توکن (Token Sequences) برآورد می‌کند.
* مدل‌های زبانی بزرگ خودرگرسیو (Autoregressive LLMs) احتمال دنباله را به پیش‌بینی‌های توکن بعدی (Next-Token Predictions) تجزیه می‌کنند.
* توکن‌سازی (Tokenization) متن را به واحدهای گسسته‌ای تبدیل می‌کند که توسط مدل پردازش می‌شوند.
* ترنسفورمرها (Transformers) از توجه (Attention) برای ساخت بازنمایی‌های زمینه‌مند توکن (Contextual Token Representations) استفاده می‌کنند.
* قابلیت مدل زبانی بزرگ (LLM Capability) به معماری، داده، محاسبات، بهینه‌سازی و پس‌آموزش بستگی دارد.
* پیش‌آموزش (Pretraining) توانایی عمومی ادامه‌دادن متن را توسعه می‌دهد؛ پس‌آموزش (Post-training) رفتار مفید را شکل می‌دهد.
* آموزش (Training)، استنتاج (Inference) و سرویس‌دهی تولیدی (Production Serving) بارهای کاری مهندسی متمایزی هستند.
* مدل‌های زبانی بزرگ مدل‌های احتمالاتی قدرتمندی هستند، اما پایگاه‌های دادهٔ واقعی تضمین‌شده نیستند.
* RAG و استفاده از ابزار (Tool Use) می‌توانند قابلیت‌های مدل را گسترش دهند، اما حالت‌های شکست اضافی نیز ایجاد می‌کنند.
* توسعهٔ قابل‌اعتماد مدل زبانی بزرگ (Reliable LLM Development) به کار هماهنگ در داده، مدل‌سازی، ارزیابی و سامانه‌ها نیاز دارد.

---

## پرسش‌های مروری (Review Questions)

1. قانون زنجیره‌ای (Chain Rule) چگونه از مدل‌سازی زبانی خودرگرسیو (Autoregressive Language Modeling) پشتیبانی می‌کند؟
2. چرا مدل‌های N-Gram به دنباله‌های دیده‌نشده به‌خوبی تعمیم نمی‌یابند؟
3. ترنسفورمرها (Transformers) چه مزیت‌هایی نسبت به مدل‌های بازگشتی (Recurrent Models) معرفی کردند؟
4. پارامترها (Parameters)، فعال‌سازی‌ها (Activations) و ابرپارامترها (Hyperparameters) چه تفاوتی با هم دارند؟
5. چرا یک مدل پایه (Base Model) به‌طور خودکار یک دستیار گفتگومحور مفید (Useful Conversational Assistant) نیست؟
6. تفاوت بین پیش‌پرکنی (Prefill) و رمزگشایی (Decode) در زمان استنتاج (Inference) چیست؟
7. چرا پرپلکسیتی پایین (Low Perplexity) ممکن است کیفیت پیروی از دستورالعمل (Instruction-following Quality) را پیش‌بینی نکند؟
8. یادگیری درون‌زمینه (In-Context Learning) چه تفاوتی با تنظیم دقیق (Fine-tuning) دارد؟
9. چرا RAG توهم (Hallucination) را به‌طور کامل حذف نمی‌کند؟
10. کدام بخش‌های چرخهٔ حیات مدل زبانی بزرگ (LLM Lifecycle) عمدتاً مسائل سامانه‌ای (Systems Problems) هستند؟

---

## مطالعهٔ بیشتر (Further Reading)

1. Bengio, Y., Ducharme, R., Vincent, P., and Jauvin, C.
   *A Neural Probabilistic Language Model*. Journal of Machine Learning Research, 2003.

2. Vaswani, A. et al.
   *Attention Is All You Need*. NeurIPS, 2017.

3. Radford, A. et al.
   *Language Models are Unsupervised Multitask Learners*. OpenAI, 2019.

4. Brown, T. et al.
   *Language Models are Few-Shot Learners*. NeurIPS, 2020.

5. Bommasani, R. et al.
   *On the Opportunities and Risks of Foundation Models*. 2021.

6. Zhao, W. X. et al.
   *A Survey of Large Language Models*. 2023.

---

[قبلی: پیشگفتار (Preface)](./00-preface.md) |
[فهرست (Contents)](./index.md) | [بعدی: داده برای آموزش مدل زبانی بزرگ (Data for LLM Training)](./02-data.md)
