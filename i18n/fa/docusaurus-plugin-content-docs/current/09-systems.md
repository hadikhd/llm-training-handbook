---
id: systems
title: سامانه‌ها
sidebar_label: سامانه‌ها
sidebar_position: 1
---
# فصل ۹: سامانه‌ها
<div className="chapter-hero">

![فصل ۹: سامانه‌ها](/img/chapters/systems.png)

</div>

[فصل قبل: ارزیابی](./08-evaluation.md) |
[فهرست مطالب](./index.md) |
[فصل بعد: استنتاج و رمزگشایی](./10-inference-and-decoding.md)

---
## ۱. چرا سیستم‌ها اهمیت دارند 

آموزش یک مدل زبانی بزرگ (large language model) فقط یک مسئله یادگیری ماشین (machine learning) نیست. این همچنین یک مسئله سیستم‌های توزیع‌شده (distributed systems) است.

هدف ریاضیاتی آموزش (training) ممکن است ساده باشد:

```text
predict the next token
```
اما اجرای کارآمد این هدف (objective) روی صدها یا هزاران شتاب‌دهنده (accelerator) بسیار پیچیده‌تر است. سیستم (system) باید به‌طور پیوسته داده (data) را بارگذاری کند، محاسبه توزیع‌شده (distributed computation) را اجرا کند، گرادیان‌ها (gradients) را همگام‌سازی کند، پارامترها (parameters) را به‌روزرسانی کند، نقطه‌های بازرسی (checkpoints) را ذخیره کند، از خرابی‌ها (failures) بازیابی شود و به‌اندازه کافی فراداده (metadata) ثبت کند تا اجرای انجام‌شده (run) بازتولید شود.

در مقیاس کوچک (small scale)، کد ناکارآمد (inefficient code) ممکن است فقط آموزش را کندتر کند. در مقیاس بزرگ (large scale)، همین ناکارآمدی می‌تواند محاسبات (compute) قابل‌توجهی را هدر دهد، باعث ناپایداری (instability) شود یا اجرای آموزش را از نظر اقتصادی غیرعملی (infeasible) کند.

یک سیستم موفق برای مدل زبانی بزرگ (LLM system) باید بین موارد زیر تعادل (balance) برقرار کند:

- درستی عددی (numerical correctness)
- پایداری آموزش (training stability)
- بهره‌برداری از سخت‌افزار (hardware utilization)
- بهره‌وری حافظه (memory efficiency)
- بهره‌وری ارتباط (communication efficiency)
- توان عملیاتی ذخیره‌سازی (storage throughput)
- تحمل‌پذیری در برابر خطا (fault tolerance)
- مشاهده‌پذیری (observability)
- بازتولیدپذیری (reproducibility)
- هزینه عملیاتی (operational cost)

هدف صرفاً این نیست که آموزش اجرا شود. هدف این است که به‌درستی، کارآمد و با قابلیت بازیابی اجرا شود.

---

## ۲. پشتهٔ آموزش LLM

یک سیستم آموزش مدل زبانی بزرگ (LLM training system) شامل چندین لایه (layer) تعاملی است:

```text
training data
-> storage and data pipeline
-> tokenizer and sequence packing
-> training framework
-> distributed runtime
-> communication libraries
-> accelerator kernels
-> GPU or TPU hardware
-> cluster network
-> checkpoint storage
-> monitoring and orchestration
```
یک مشکل (problem) در هر لایه (layer) می‌تواند کل سیستم (system) را محدود کند.

نمونه‌ها (examples) شامل موارد زیر هستند:

- ذخیره‌سازی کند (slow storage) که باعث می‌شود شتاب‌دهنده‌ها (accelerators) منتظر داده (data) بمانند
- ماسک‌گذاری نادرست زیان (incorrect loss masking) که هدف (objective) را خراب می‌کند
- بسته‌بندی ضعیف توالی (poor sequence packing) که محاسبات (compute) را روی پدینگ (padding) هدر می‌دهد
- سربار ارتباط (communication overhead) که مقیاس‌پذیری توزیع‌شده (distributed scaling) را محدود می‌کند
- نوشتن نقطه بازرسی (checkpoint write) که آموزش را متوقف می‌کند
- کرنل‌های ناپایدار (unstable kernels) که `NaN` تولید می‌کنند
- خرابی زمان‌بند (scheduler failure) که کارگرهای سالم (healthy workers) را خاتمه می‌دهد
- ثبت وقایع ناکافی (insufficient logging) که تشخیص خرابی (failure diagnosis) را غیرممکن می‌کند

بنابراین عملکرد سیستم (system performance) باید به‌صورت سرتاسری (end to end) تحلیل شود.

---

## ۳. مبانی سخت‌افزار

بیشتر مدل‌های زبانی بزرگ مدرن (modern LLMs) روی GPU یا دیگر شتاب‌دهنده‌های تخصصی (specialized accelerators) آموزش داده می‌شوند.

ویژگی‌های مهم سخت‌افزار (hardware characteristics) شامل موارد زیر هستند:

| ویژگی (Characteristic) | چرا مهم است (Why It Matters) |
|---|---|
| توان عملیاتی محاسباتی (compute throughput) | نرخ عملیات ماتریسی (matrix operations) را تعیین می‌کند |
| حافظه با پهنای باند بالا (high-bandwidth memory) | پارامترها (parameters)، گرادیان‌ها (gradients)، فعال‌سازی‌ها (activations) و حالت‌های بهینه‌ساز (optimizer states) را ذخیره می‌کند |
| پهنای باند حافظه (memory bandwidth) | تعیین می‌کند که داده (data) با چه سرعتی به واحد محاسبه (compute unit) منتقل می‌شود |
| پهنای باند اتصال داخلی (interconnect bandwidth) | سرعت ارتباط توزیع‌شده (distributed communication) را تعیین می‌کند |
| دقت پشتیبانی‌شده (supported precision) | `FP32`، `TF32`، `FP16`، `BF16`، `FP8` یا دیگر قالب‌ها (formats) را ممکن می‌کند |
| پشتیبانی کرنل (kernel support) | مشخص می‌کند آیا عملیات بهینه‌شده (optimized operations) در دسترس هستند یا نه |
| قابلیت‌های اطمینان‌پذیری (reliability features) | به تشخیص و بازیابی از خطاهای سخت‌افزاری (hardware errors) کمک می‌کنند |

مشخصات اوج سخت‌افزار (peak hardware specifications) نظری هستند. توان عملیاتی واقعی آموزش (real training throughput) معمولاً کمتر است، چون جابه‌جایی داده (data movement)، ارتباط (communication)، فراخوانی کرنل (kernel launch)، همگام‌سازی (synchronization)، بارگذاری داده (data loading) و عدم‌تعادل خط لوله (pipeline imbalance) وجود دارد.

پرسش مفید این است:

چه مقدار از ظرفیت نظری سخت‌افزار (theoretical hardware capacity) توسط بارکاری آموزش (training workload) حفظ می‌شود؟

---

## ۴. عملیات محاسبه‌محور و حافظه‌محور

یک عملیات (operation) زمانی وابسته به محاسبه (compute-bound) است که توان عملیاتی حسابی (arithmetic throughput) محدودیت اصلی باشد. زمانی وابسته به حافظه (memory-bound) است که جابه‌جایی داده (data movement) محدودیت اصلی باشد.

ضرب‌های ماتریسی بزرگ (large matrix multiplications) اغلب زمانی وابسته به محاسبه (compute-bound) هستند که ابعادشان (dimensions) به‌اندازه کافی بزرگ باشد. عملیاتی مانند نرمال‌سازی (normalization)، تابع فعال‌سازی (activation function)، بازیابی تعبیه (embedding lookup) و به‌روزرسانی بهینه‌ساز (optimizer update) ممکن است با پهنای باند حافظه (memory bandwidth) محدود شوند.

این تمایز (distinction) مهم است چون بهینه‌سازی‌های مختلف (different optimizations) گلوگاه‌های متفاوت (different bottlenecks) را هدف می‌گیرند:

| گلوگاه (Bottleneck) | بهینه‌سازی بالقوه (Potential Optimization) |
|---|---|
| توان عملیاتی محاسباتی (compute throughput) | دقت پایین‌تر (lower precision)، کرنل ماتریسی بهینه‌شده (optimized matrix kernel) |
| پهنای باند حافظه (memory bandwidth) | همجوشی کرنل (kernel fusion)، کاهش خواندن/نوشتن حافظه (less memory read/write) |
| ظرفیت حافظه GPU (GPU memory capacity) | خردسازی (sharding)، نقطه‌بازرسی‌گذاری (checkpointing)، ریز-بچ‌های کوچک‌تر (smaller micro-batches) |
| ارتباط (communication) | چیدمان بهتر موازی‌سازی (better parallelism layout)، هم‌پوشانی (overlap)، اتصال داخلی سریع‌تر (faster interconnect) |
| بارگذاری داده (data loading) | پیش‌واکشی (prefetching)، کش‌گذاری (caching)، مجموعه‌داده خردشده (sharded dataset) |
| همگام‌سازی (synchronization) | کاهش سدها (fewer barriers) و کاهش عدم‌تعادل بار (less load imbalance) |

اضافه‌کردن شتاب‌دهنده‌های بیشتر (more accelerators)، یک گلوگاه پهنای باند حافظه (memory-bandwidth bottleneck) یا گلوگاه خط لوله داده (data-pipeline bottleneck) را به‌صورت خودکار حل نمی‌کند.

---

## ۵. مصرف حافظه در حین آموزش

حافظه آموزش (training memory) فقط توسط پارامترهای مدل (model parameters) مصرف نمی‌شود.

مولفه‌های اصلی (main components) عبارت‌اند از:

```text
parameters
+ gradients
+ optimizer states
+ activations
+ temporary buffers
+ communication buffers

```
برای Adam، حالت‌های بهینه‌ساز (optimizer states) معمولاً شامل ممان اول (first moment) و برآورد ممان دوم (second-moment estimate) هستند. بسته به دقت (precision) و پیاده‌سازی (implementation)، حالت‌های بهینه‌ساز (optimizer states) و گرادیان‌ها (gradients) می‌توانند به‌مراتب حافظه بیشتری از خود پارامترها (parameters) نیاز داشته باشند.

یک تقسیم‌بندی مفهومی تقریبی (rough conceptual breakdown) به شکل زیر است:

| مولفه (Component) | عمدتاً با چه چیزی مقیاس می‌شود (Scales Mainly With) |
|---|---|
| پارامترها (parameters) | اندازه مدل (model size) |
| گرادیان‌ها (gradients) | اندازه مدل (model size) |
| حالت‌های بهینه‌ساز (optimizer states) | اندازه مدل (model size) |
| فعال‌سازی‌ها (activations) | اندازه بچ (batch size)، طول توالی (sequence length)، تعداد لایه‌ها (layers)، اندازه نهان (hidden size) |
| میانی‌های توجه (attention intermediates) | اندازه بچ (batch size)، تعداد هدها (heads)، طول توالی (sequence length) |
| بافرهای ارتباط (communication buffers) | راهبرد توزیع‌شده (distributed strategy) و اندازه سطل (bucket size) |

برآورد حافظه (memory estimate) باید تخصیص موقت (temporary allocation) و تکه‌تکه‌شدن (fragmentation) را هم در نظر بگیرد. پیکربندی‌ای که بر اساس یک محاسبه ساده‌شده (simplified calculation) مناسب به نظر می‌رسد، ممکن است همچنان در زمان اجرا (runtime) شکست بخورد.

---

## ۶. آموزش با دقت ترکیبی (Mixed-Precision Training)

آموزش با دقت ترکیبی (mixed-precision training) از قالب‌های عددی با دقت پایین‌تر (lower-precision numeric formats) برای بسیاری از عملیات‌ها (operations) استفاده می‌کند، در حالی که دقت بالاتر (higher precision) را در جاهایی که برای پایداری (stability) لازم است حفظ می‌کند.

قالب‌های رایج (common formats) شامل موارد زیر هستند:

| قالب (Format) | ویژگی‌ها (Characteristics) |
|---|---|
| `FP32` | دقت (precision) و بازه (range) بالا، هزینه زیاد حافظه (memory) و محاسبه (compute) |
| `TF32` | عملیات ماتریسی سریع‌تر (faster matrix operations) روی سخت‌افزارهای پشتیبانی‌شده (supported hardware) |
| `FP16` | مصرف کم حافظه (low memory use)، بازه عددی محدود (limited numerical range) |
| `BF16` | مصرف کم حافظه (low memory use) با بازه توان وسیع‌تر (wider exponent range) نسبت به `FP16` |
| `FP8` | بهره‌وری بیشتر (higher efficiency)، اما نیازمند مقیاس‌دهی دقیق (careful scaling) و پشتیبانی سخت‌افزاری (hardware support) |

`BF16` به‌طور گسترده برای آموزش مدل زبانی بزرگ (LLM training) استفاده می‌شود، چون تعادل مفیدی بین بهره‌وری (efficiency) و پایداری عددی (numerical stability) ارائه می‌دهد.

دقت ترکیبی (mixed precision) همچنان به توجه نسبت به موارد زیر نیاز دارد:

- وزن اصلی (master weight)
- دقت حالت بهینه‌ساز (optimizer-state precision)
- دقت انباشت گرادیان (gradient accumulation precision)
- مقیاس‌دهی زیان (loss scaling) برای `FP16`
- تشخیص سرریز (overflow detection) و کم‌ریزی (underflow detection)
- عملیات‌های حساس (sensitive operations) مانند نرمال‌سازی (normalization) و کاهش (reduction)

دقت پایین‌تر (lower precision) فقط زمانی بهره‌وری (efficiency) را بهبود می‌دهد که کل پشته آموزش (training stack) آن را به‌درستی پشتیبانی کند.

---

## ۷. آموزش توزیع‌شده (Distributed Training)

وقتی مدل (model) یا بارکاری آموزش (training workload) دیگر به‌صورت کارآمد روی یک شتاب‌دهنده (accelerator) جا نمی‌شود، کار باید توزیع شود (distributed).

شکل‌های اصلی موازی‌سازی (parallelism) عبارت‌اند از:

- موازی‌سازی داده (data parallelism)
- موازی‌سازی تنسور (tensor parallelism)
- موازی‌سازی خط لوله (pipeline parallelism)
- موازی‌سازی توالی یا بافت (sequence or context parallelism)
- موازی‌سازی خبره (expert parallelism)

این روش‌ها (methods) می‌توانند به موازی‌سازی چندبُعدی (multi-dimensional parallelism) ترکیب شوند.

برای مثال (for example):

```text
total_accelerators =
data_parallel_size
* tensor_parallel_size
* pipeline_parallel_size
* context_parallel_size

```
راهبرد درست (correct strategy) به اندازه مدل (model size)، طول توالی (sequence length)، حافظه سخت‌افزار (hardware memory)، توپولوژی شبکه (network topology)، اندازه بچ (batch size) و بلوغ پیاده‌سازی (implementation maturity) بستگی دارد.

---

## ۸. آموزش توزیع‌شده (Distributed Training)

موازی‌سازی داده (data parallelism) یک کپی از مدل (model copy) را روی هر کارگر (worker) قرار می‌دهد و به هر کارگر بخش متفاوتی از بچ (batch) را می‌دهد.

فرایند پایه (basic process) به این صورت است:

```text
each worker:
runs forward pass
computes local gradients

all workers:
synchronize gradients
apply equivalent parameter updates

```
اندازه بچ سراسری مؤثر (effective global batch size) تقریباً برابر است با:

```text
global_batch_size =
micro_batch_size
* gradient_accumulation_steps
* data_parallel_workers

```
موازی‌سازی داده استاندارد (standard data parallelism) زمانی ساده و مؤثر است که مدل (model) روی هر دستگاه (device) جا شود. محدودیت اصلی آن این است که پارامترها (parameters)، گرادیان‌ها (gradients) و حالت‌های بهینه‌ساز (optimizer states) روی کارگرها (workers) تکثیر می‌شوند (replicated).

همگام‌سازی گرادیان (gradient synchronization) معمولاً از عملیات جمعی `all-reduce` (collective operation) استفاده می‌کند. در مقیاس بزرگ (large scale)، هزینه این ارتباط (communication cost) می‌تواند قابل‌توجه شود.

---

## ۹. موازی‌سازی کاملِ شاردشدهٔ داده (FSDP)

موازی‌سازی داده کاملاً خردشده (fully sharded data parallelism) با خردسازی حالت‌های مدل (model states) میان کارگرهای موازی‌سازی داده (data-parallel workers)، تکثیر حافظه (memory replication) را کاهش می‌دهد.

بسته به پیاده‌سازی (implementation)، ممکن است موارد زیر را خرد کند (shard):

- حالت‌های بهینه‌ساز (optimizer states)
- گرادیان‌ها (gradients)
- پارامترها (parameters)

این خانواده از راهبردها (family of strategies) شامل مفاهیمی است که معمولاً با `ZeRO` و موازی‌سازی داده کاملاً خردشده (Fully Sharded Data Parallel) یا `FSDP` مرتبط هستند.

یک پیشروی ساده‌شده (simplified progression) به این صورت است:

| راهبرد (Strategy) | چه چیزی خرد می‌شود (What Is Sharded) |
|---|---|
| موازی‌سازی داده پایه (basic data parallelism) | هیچ‌چیز (nothing) |
| خردسازی بهینه‌ساز (optimizer sharding) | حالت‌های بهینه‌ساز (optimizer states) |
| خردسازی گرادیان (gradient sharding) | حالت‌های بهینه‌ساز (optimizer states) و گرادیان‌ها (gradients) |
| خردسازی کامل (full sharding) | حالت‌های بهینه‌ساز (optimizer states)، گرادیان‌ها (gradients) و پارامترها (parameters) |

خردسازی کامل (full sharding) مصرف حافظه را کاهش می‌دهد، اما ارتباط اضافی (extra communication) ایجاد می‌کند. ممکن است پارامترها (parameters) پیش از محاسبه (computation) گردآوری شوند (gathered) و پس از آن آزاد (released) یا دوباره خرد شوند (resharded).

گزینه‌های مهم پیکربندی (configuration options) شامل موارد زیر هستند:

- دانه‌بندی خردسازی (sharding granularity)
- سیاست پیچش (wrapping policy)
- رفتار پیش‌واکشی (prefetch behavior)
- اندازه سطل ارتباط (communication bucket size)
- سیاست دقت ترکیبی (mixed-precision policy)
- تخلیه به CPU (CPU offloading)
- قالب نقطه بازرسی (checkpoint format)

تنظیمات `FSDP` باید پیش از یک اجرای بزرگ آموزش (large training run)، در مقیاس کوچک (small scale) اعتبارسنجی شوند.

---

## ۱۰. موازی‌سازی تنسوری (Tensor Parallelism)

موازی‌سازی تنسور (tensor parallelism) عملیات‌های تنسوری منفرد (individual tensor operations) را بین چند شتاب‌دهنده (accelerators) تقسیم می‌کند.

برای مثال (for example)، یک لایه خطی بزرگ (large linear layer) ممکن است بر اساس سطر (row) یا ستون (column) بخش‌بندی شود:

```text
Y = X`W`
```
ماتریس وزن (weight matrix) `W` بین دستگاه‌ها (devices) تقسیم می‌شود (split) و کارگرها (workers) برای محاسبه نتیجه کامل (complete result) همکاری می‌کنند.

موازی‌سازی تنسور (tensor parallelism) زمانی مفید است که یک لایه (layer) یا مدل (model) نتواند به‌صورت کارآمد روی یک شتاب‌دهنده (accelerator) جا شود. این روش معمولاً به ارتباط مکرر (frequent communication) درون هر لایه ترنسفورمر (Transformer layer) نیاز دارد، بنابراین از اتصال داخلی سریع و کم‌تأخیر (fast, low-latency interconnect) سود می‌برد.

گروه‌های موازی تنسور (tensor-parallel groups) معمولاً درون دامنه‌های سخت‌افزاری با اتصال فشرده (tightly connected hardware domains) نگه داشته می‌شوند، مانند شتاب‌دهنده‌هایی (accelerators) که درون یک گره (node) با پیوندهای پرپهنای‌باند (high-bandwidth links) به هم متصل‌اند.

قرارگیری نامناسب کارگرهای موازی تنسور (poor placement of tensor-parallel workers) می‌تواند باعث شود ارتباط (communication) بر محاسبه (computation) غلبه کند.

---

## ۱۱. موازی‌سازی خط لوله‌ای (Pipeline Parallelism)

موازی‌سازی خط لوله (pipeline parallelism) گروه‌های متفاوتی از لایه‌های ترنسفورمر (Transformer layers) را به دستگاه‌ها (devices) یا مراحل مختلف (stages) اختصاص می‌دهد.

یک بچ (batch) به ریز-بچ‌ها (micro-batches) تقسیم می‌شود که از مراحل (stages) عبور می‌کنند:

```text
stage 1 -> stage 2 -> stage 3 -> stage 4

```
بدون ریز-بچ‌بندی (micro-batching)، بیشتر مراحل (stages) در حالی بیکار (idle) می‌مانند که فقط یک مرحله فعال است. زمان‌بندی‌های خط لوله (pipeline schedules) با پردازش هم‌زمان چندین ریز-بچ (micro-batch)، بهره‌برداری (utilization) را بهبود می‌دهند.

موازی‌سازی خط لوله (pipeline parallelism) چند چالش (challenges) ایجاد می‌کند:

- حباب خط لوله (pipeline bubble)
- عدم‌تعادل مرحله (stage imbalance)
- انتقال فعال‌سازی (activation transfer)
- پیچیدگی زمان‌بندی (scheduling complexity)
- برهم‌کنش با انباشت گرادیان (interaction with gradient accumulation)
- نقطه‌بازرسی‌گذاری پیچیده‌تر (more complex checkpointing)
- حساسیت نسبت به کارگر کند (sensitivity to a slow worker)

لایه‌ها (layers) باید طوری اختصاص داده شوند (assigned) که مراحل (stages) تقریباً محاسبه (computation) و نیاز حافظه (memory requirement) متعادلی داشته باشند. تعداد برابر لایه‌ها (equal numbers of layers) همیشه بارکاری برابر (equal workload) تولید نمی‌کند.

---

## ۱۲. موازی‌سازی توالی و زمینه (Sequence and Context Parallelism)

توالی‌های بلند (long sequences) می‌توانند فعال‌سازی‌ها (activations) و محاسبه توجه (attention computation) را برای یک شتاب‌دهنده منفرد (single accelerator) بیش از حد بزرگ کنند.

موازی‌سازی توالی (sequence parallelism) عملیات‌ها (operations) را در امتداد بُعد توالی (sequence dimension) بخش‌بندی می‌کند (partition). موازی‌سازی بافت (context parallelism) پردازش بافت بلند (long-context processing) را بین دستگاه‌ها (devices) توزیع می‌کند و به هر کارگر (worker) اجازه می‌دهد بخشی از بافت (context) را مدیریت کند.

این روش‌ها (methods) می‌توانند حافظه فعال‌سازی هر دستگاه (per-device activation memory) را کاهش دهند، اما توجه (attention) به تبادل اطلاعات (information exchange) در بین بخش‌بندی‌های توالی (sequence partitions) نیاز دارد.

طراحی (design) باید موارد زیر را در نظر بگیرد:

- ماسک‌گذاری علّی (causal masking)
- کدگذاری مکانی (positional encoding)
- تبادل کلید-مقدار (key-value exchange)
- حجم ارتباط (communication volume)
- متعادل‌سازی بار (load balancing)
- سازگاری با کرنل توجه بهینه‌شده (compatibility with optimized attention kernels)

آموزش بافت بلند (long-context training) به همان اندازه که یک چالش معماری (architectural challenge) است، یک چالش سیستمی (systems challenge) نیز هست.

---

## ۱۳. موازی‌سازی متخصصان (Expert Parallelism)

مدل‌های ترکیب خبرگان (Mixture-of-Experts models) چندین خبره پیش‌خور (feed-forward experts) دارند، در حالی که برای هر توکن (token) فقط زیرمجموعه‌ای از آن‌ها فعال می‌شود.

یک فرایند ساده‌شده (simplified process) به این صورت است:

```text
token representation
-> router
-> selected experts
-> weighted expert outputs
```
موازی‌سازی خبره (expert parallelism) خبره‌ها (experts) را بین کارگرها (workers) توزیع می‌کند.

این کار محاسبه فعال (active compute) را نسبت به تعداد کل پارامترها (total parameter count) کاهش می‌دهد، اما چالش‌های سیستمی جدیدی (new systems challenges) ایجاد می‌کند:

- ارتباط مسیریابی توکن (token routing communication)
- بهره‌برداری نامتوازن از خبره‌ها (uneven expert utilization)
- خبره‌های بیش‌بار (overloaded experts)
- محدودیت ظرفیت (capacity limit)
- حذف یا بازمسیریابی توکن‌ها (dropped or rerouted tokens)
- ارتباط `all-to-all`
- جانمایی خبره (expert placement)
- پایداری مسیریابی (routing stability)

یک مسیریاب (router) با تعادل ضعیف (poor balance) می‌تواند بعضی دستگاه‌ها (devices) را بیش‌بار (overloaded) و بعضی دیگر را کم‌استفاده (underused) بگذارد.

---

## ۱۴. موازی‌سازی سه‌بعدی و ترکیبی (3D and Hybrid Parallelism)

اجراهای بزرگ آموزش (large training runs) اغلب چندین راهبرد موازی‌سازی (parallelism strategies) را ترکیب می‌کنند.

یک طراحی رایج (common design) این است:

```text
data parallelism
+ tensor parallelism
+ pipeline parallelism

```

موازی‌سازی اضافی توالی (sequence parallelism)، بافت (context parallelism) یا خبره (expert parallelism) ممکن است برای بافت بلند (long context) یا مدل‌های MoE اضافه شود.

چیدمان موازی‌سازی (parallelism layout) باید توپولوژی فیزیکی خوشه (physical cluster topology) را بازتاب دهد:

```text
fastest links:
tensor parallelism

high-bandwidth local or nearby links:
pipeline or expert communication

larger cross-node groups:
data parallelism

```
این یک قاعده همگانی (universal rule) نیست، اما عملیات‌های پرترافیک ارتباطی (communication-heavy operations) عموماً باید از سریع‌ترین پیوندهای موجود (fastest available links) استفاده کنند.

ابعاد بیشتر موازی‌سازی (more parallelism dimensions)، پیچیدگی پیاده‌سازی (implementation complexity) و پیچیدگی عملیاتی (operational complexity) را افزایش می‌دهند. کوچک‌ترین پیکربندی (smallest configuration) که حافظه (memory) و توان عملیاتی (throughput) لازم را برآورده کند، معمولاً اعتبارسنجی و نگه‌داری آسان‌تری دارد.

---

## ۱۵. عملیات جمعی ارتباطی (Communication Collectives)

آموزش توزیع‌شده (distributed training) به عملیات‌های جمعی ارتباطی (collective communication operations) متکی است.

عملیات‌های جمعی رایج (common collectives) شامل موارد زیر هستند:

| عملیات جمعی (Collective) | هدف (Purpose) |
|---|---|
| All-reduce | تجمیع مقدارها (aggregate values) در سراسر کارگرها (workers) |
| All-gather | گردآوری خرده‌ها (collect shards) از همه کارگرها (workers) |
| Reduce-scatter | تجمیع و توزیع خرده‌ها (aggregate and distribute shards) |
| Broadcast | ارسال داده (send data) از یک کارگر (worker) به بقیه |
| All-to-all | تبادل داده‌های متفاوت (exchange different data) میان همه کارگرها (workers) |

راهبردهای مختلف موازی‌سازی (different parallelism strategies) الگوهای ارتباطی متفاوتی (different communication patterns) تولید می‌کنند.

عملکرد ارتباط (communication performance) به موارد زیر بستگی دارد:

- اندازه پیام (message size)
- پهنای باند شبکه (network bandwidth)
- تأخیر شبکه (network latency)
- توپولوژی (topology)
- پیاده‌سازی عملیات جمعی (collective implementation)
- جانمایی فرایند (process placement)
- رقابت (contention)
- هم‌پوشانی با محاسبه (overlap with computation)

ارتباط کوچک و پرتکرار (small and frequent communication) می‌تواند وابسته به تأخیر (latency-bound) باشد. انتقال‌های بزرگ (large transfers) بیشتر احتمال دارد وابسته به پهنای باند (bandwidth-bound) باشند.

---

## ۱۶. هم‌پوشانی ارتباطات و محاسبات

آموزش توزیع‌شده (distributed training) زمانی کارآمدتر می‌شود که ارتباط (communication) هم‌زمان با محاسبه مفید (useful computation) رخ دهد.

نمونه‌ها (examples) شامل موارد زیر هستند:

- همگام‌سازی یک سطل گرادیان (gradient bucket) در حالی که سطل دیگری (another bucket) در حال محاسبه است
- پیش‌واکشی پارامترها (prefetching parameters) برای لایه بعدی (next layer)
- بارگذاری بچ داده بعدی (next data batch) در طول گام فعلی (current step)
- نوشتن ناهمگام خرده‌های نقطه بازرسی (asynchronous checkpoint shards)
- هم‌پوشانی دادن انتقال خط لوله (pipeline transfer) با محاسبه مرحله (stage computation)

هم‌پوشانی (overlap) خودکار نیست. ممکن است توسط همگام‌سازی جریان (stream synchronization)، ترتیب وابستگی (dependency ordering)، فشار حافظه (memory pressure) یا رقابت بر سر منبع (resource contention) محدود شود.

برای راستی‌آزمایی (verification) اینکه ارتباط واقعاً پنهان شده است (hidden) و فقط به‌صورت هم‌زمان زمان‌بندی نشده (concurrently scheduled)، به پروفایل‌گر (profiler) نیاز است.

---

## ۱۷. مکانیزم Attention کارآمد

توجه استاندارد (standard attention) نسبت به طول توالی (sequence length) پیچیدگی درجه دوم (quadratic complexity) دارد:

```text
attention_compute ∝ sequence_length^2
```
پیاده‌سازی‌های ساده‌لوحانه (naive implementations) ممکن است ماتریس توجه میانی بزرگی (large intermediate attention matrix) را نیز ایجاد کنند (materialize).

کرنل‌های توجه با حافظه‌کارآمد (memory-efficient attention kernels)، مانند پیاده‌سازی‌های سبک FlashAttention، محاسبه را بازسازمان‌دهی می‌کنند (reorganize computation) تا ترافیک حافظه با پهنای باند بالا (high-bandwidth memory traffic) را کاهش دهند و از ذخیره میانی‌های غیرضروری (unnecessary intermediates) اجتناب کنند.

مزیت‌ها (benefits) ممکن است شامل موارد زیر باشد:

- حافظه فعال‌سازی کمتر (less activation memory)
- توان عملیاتی بالاتر (higher throughput)
- پشتیبانی از توالی‌های بلندتر (support for longer sequences)
- کاهش خواندن/نوشتن حافظه (less memory read/write)

درستی (correctness) همچنان به پشتیبانی از موارد زیر وابسته است:

- ماسک علّی (causal mask)
- ماسک پدینگ (padding mask)
- طول توالی متغیر (variable sequence length)
- dropout
- توجه با پرس‌وجوی گروه‌بندی‌شده (grouped-query attention)
- توجه پنجره لغزان (sliding-window attention)
- دقت عددی موردنیاز (required numerical precision)

یک کرنل بهینه‌شده (optimized kernel) باید در برابر یک پیاده‌سازی مرجع مورداعتماد (trusted reference implementation) تست شود.

---

## ۱۸. ادغام Kernel و کامپایل

یک ترنسفورمر (Transformer) شامل عملیات‌های زیادی (many operations) است که در غیر این صورت ممکن است به فراخوانی‌های جداگانه کرنل (separate kernel launches) و دسترسی تکراری به حافظه (repeated memory access) نیاز داشته باشند.

همجوشی کرنل (kernel fusion) عملیات‌های سازگار (compatible operations) را ترکیب می‌کند، مانند:

- افزودن بایاس و فعال‌سازی (bias addition and activation)
- نرمال‌سازی و افزودن باقیمانده (normalization and residual addition)
- عملیات به‌روزرسانی بهینه‌ساز (optimizer update operations)
- اعمال موقعیت چرخشی (rotary-position application)
- عملیات مرتبط با softmax

سامانه کامپایل (compilation system) همچنین می‌تواند عملیات‌ها را همجوش کند (fuse operations)، گراف (graph) را برای شکل‌های شناخته‌شده (known shapes) تخصصی‌سازی کند (specialize) و سربار مفسر (interpreter overhead) را کاهش دهد.

مزیت‌های بالقوه (potential benefits) شامل موارد زیر هستند:

- فراخوانی کرنل کمتر (fewer kernel launches)
- ترافیک حافظه کمتر (less memory traffic)
- بهره‌برداری بیشتر از شتاب‌دهنده (higher accelerator utilization)
- توان عملیاتی بهتر (better throughput)

ریسک‌های بالقوه (potential risks) شامل موارد زیر هستند:

- زمان کامپایل طولانی (long compilation time)
- شکست گراف (graph break)
- کامپایل مجدد وابسته به شکل (shape-dependent recompilation)
- تفاوت عددی (numerical difference)
- اشکال‌زدایی دشوار (difficult debugging)
- رفتار وابسته به سخت‌افزار (hardware-specific behavior)

افزایش عملکرد (performance gain) باید با طول توالی (sequence length) و اندازه بچ (batch size) نماینده اندازه‌گیری شود.

---

## ۱۹. Checkpointing فعال‌سازی‌ها (Activation Checkpointing)

فعال‌سازی‌های حاصل از گذر رو به جلو (forward pass activations) معمولاً برای استفاده در پس‌انتشار (backpropagation) ذخیره می‌شوند. برای مدل‌های بزرگ (large models) یا توالی‌های بلند (long sequences)، حافظه فعال‌سازی (activation memory) می‌تواند به یک قید اصلی (major constraint) تبدیل شود.

نقطه‌بازرسی‌گذاری فعال‌سازی (activation checkpointing) فقط فعال‌سازی‌های انتخابی (selected activations) را ذخیره می‌کند و مقادیر میانی ازدست‌رفته (lost intermediate values) را در گذر رو به عقب (backward pass) دوباره محاسبه می‌کند (recompute).

مبادله (trade-off) به این صورت است:

```text
less memory
in exchange for
more computation

```

نقطه‌بازرسی‌گذاری (checkpointing) می‌تواند بچ بزرگ‌تر (larger batch)، توالی بلندتر (longer sequence) یا مدل بزرگ‌تر (larger model) را ممکن کند. با این حال، محاسبه مجدد تهاجمی (aggressive recomputation) ممکن است توان عملیاتی (throughput) را به‌شدت کاهش دهد.

گزینه‌های مفید پیکربندی (useful configuration options) شامل موارد زیر هستند:

- نقطه بازرسی گرفتن از هر بلوک ترنسفورمر (checkpointing each Transformer block)
- نقطه بازرسی گرفتن از زیرماژول‌های انتخابی (checkpointing selected submodules)
- استفاده از پیاده‌سازی‌های غیربازگشتی (non-reentrant implementations) در صورت پشتیبانی
- فقط حفظ حالت‌های گران یا ضروری (keeping only expensive or necessary states)
- هماهنگ‌سازی نقطه‌بازرسی‌گذاری با خردسازی (coordinating checkpointing with sharding)

بهترین سیاست (best policy) به این بستگی دارد که آیا بارکاری (workload) با حافظه محدود شده (memory-bound) یا با محاسبه (compute-bound).

---

## ۲۰. انتقال پردازش و داده به CPU و فضای ذخیره‌سازی (Offloading)

وقتی حافظه شتاب‌دهنده (accelerator memory) کافی نیست، پارامترها (parameters)، حالت‌های بهینه‌ساز (optimizer states) یا فعال‌سازی‌ها (activations) ممکن است به حافظه CPU (CPU memory) یا ذخیره‌سازی (storage) منتقل شوند.

تخلیه (offloading) می‌تواند آموزش مدل‌های بزرگ‌تر (training larger models) را ممکن کند، اما انتقال داده (data transfer) ممکن است به یک گلوگاه شدید (severe bottleneck) تبدیل شود.

محدودیت‌های بالقوه (potential limitations) شامل موارد زیر هستند:

- پهنای باند PCIe (PCIe bandwidth)
- پهنای باند حافظه CPU (CPU memory bandwidth)
- تأخیر ذخیره‌سازی (storage latency)
- تأخیر همگام‌سازی (synchronization delay)
- پیچیدگی بیشتر پیاده‌سازی (greater implementation complexity)

تخلیه (offloading) زمانی بیشترین فایده را دارد که انتقال‌ها (transfers) بتوانند پیش‌واکشی شوند (prefetched) و با محاسبه (computation) هم‌پوشانی داشته باشند (overlap). نباید با آن مثل حافظه رایگان (free memory) رفتار شود.

---

## ۲۱. مهندسی خط لولهٔ داده

اگر خط لوله ورودی (input pipeline) نتواند بچ‌های توکن (token batches) را با سرعت کافی تأمین کند، شتاب‌دهنده‌ها (accelerators) نمی‌توانند کارآمد (efficient) بمانند.

یک خط لوله داده آموزشی (training data pipeline) ممکن است شامل موارد زیر باشد:

```text
document storage
-> shard selection
-> reading and decompression
-> tokenization or token loading
-> filtering
-> sequence construction
-> packing
-> batching
-> host-to-device transfer

```

برای اجراهای بزرگ (large runs)، توکن‌سازی (tokenization) معمولاً پیش از آموزش (training) انجام می‌شود. مجموعه‌داده از پیش توکن‌شده (pre-tokenized dataset) کار CPU را کاهش می‌دهد و جریان آموزش (training stream) را بازتولیدپذیرتر می‌کند.

ویژگی‌های مهم خط لوله داده (important data pipeline properties) شامل موارد زیر هستند:

- درهم‌ریزی قطعی (deterministic shuffling)
- خرده‌های متعادل (balanced shards)
- دسترسی ترتیبی به ذخیره‌سازی (sequential storage access)
- پیش‌واکشی ناهمگام (asynchronous prefetching)
- حافظه میزبان سنجاق‌شده (pinned host memory)
- مدیریت خطای کارگر (worker fault handling)
- تکرار قابل‌بازیابی (restartable iteration)
- ردیابی نسخه مجموعه‌داده (dataset-version tracking)

زمان بارگذار داده (data-loader time) و زمان بیکاری شتاب‌دهنده (accelerator idle time) باید مستقیماً پایش شوند.

---

## ۲۲. بهره‌وری در بسته‌بندی توالی‌ها

سندهای با طول متغیر (variable-length documents) می‌توانند اتلاف پدینگ قابل‌توجهی (significant padding waste) ایجاد کنند.

بسته‌بندی (packing)، چندین سند (documents) را در یک توالی آموزشی با طول ثابت (fixed-length training sequence) قرار می‌دهد:

```text
[document A][EOS][document B][EOS][document C][padding]
```
بهره‌وری بسته‌بندی (packing efficiency) را می‌توان به این صورت اندازه‌گیری کرد:

```text
packing_efficiency =
non_padding_tokens / total_sequence_capacity
```
بهره‌وری بالای بسته‌بندی (high packing efficiency)، توکن‌های مفید بر ثانیه (useful tokens per second) را بهبود می‌دهد. با این حال، بسته‌بندی (packing) باید مرزهای سند (document boundaries) را حفظ کند و توجه (attention) یا زیان (loss) را به‌درستی اعمال کند.

پرسش‌های طراحی (design questions) شامل موارد زیر هستند:

- آیا توکن‌ها (tokens) می‌توانند از مرز سندها (document boundaries) عبور کرده و توجه (attention) داشته باشند؟
- آیا توکن‌های جداکننده (separator tokens) به‌درستی اضافه می‌شوند؟
- آیا برچسب‌ها (labels) به‌درستی شیفت می‌شوند (shifted)؟
- آیا زیان (loss) روی توکن‌های جداکننده (separator tokens) اعمال می‌شود؟
- آیا توالی‌های ناقص‌پرشده (partially filled sequences) نگه داشته می‌شوند؟
- آیا ترتیب بسته‌بندی (packing order) قطعی (deterministic) است؟

یک باگ بسته‌بندی (packing bug) می‌تواند هدف آموزش (training objective) را به‌صورت پنهان (silently) خراب کند، در حالی که توان عملیاتی (throughput) سالم به نظر می‌رسد.

---

## ۲۳. اندازهٔ Batch سراسری و انباشت گرادیان

بچ سراسری (global batch) بین دستگاه‌ها (devices) و گام‌های بهینه‌ساز (optimizer steps) توزیع می‌شود.

یک فرمول رایج (common formula) به این صورت است:

```text
global_batch_tokens =
micro_batch_size
* sequence_length
* gradient_accumulation_steps
* data_parallel_size
```
برای مثال‌های با طول متغیر (variable-length examples) یا بسته‌بندی‌شده (packed examples)، باید توکن‌های غیرپدینگ واقعی (actual non-padding tokens) نیز اندازه‌گیری شوند.

انباشت گرادیان (gradient accumulation) یک بچ سراسری بزرگ (large global batch) را بدون ذخیره کل بچ (entire batch) به‌صورت هم‌زمان ممکن می‌کند. هر کارگر (worker) چندین ریز-بچ (micro-batches) را پردازش می‌کند، پیش از آن‌که به‌روزرسانی بهینه‌ساز (optimizer update) اعمال شود.

نگرانی‌های مهم (important concerns) شامل موارد زیر هستند:

- نرمال‌سازی درست زیان (correct loss normalization)
- همگام‌سازی فقط در مرز انباشت موردنظر (synchronization only at the intended accumulation boundary)
- تعداد توکن سازگار (consistent token count) در سراسر کارگرها (workers)
- مدیریت طول توالی متغیر (handling variable sequence length)
- تنظیم نرخ یادگیری (learning-rate adjustment) پس از تغییر بچ (batch)
- جلوگیری از کاهش ناخواسته گرادیان (preventing unintended gradient reduction) در هر ریز-گام (micro-step)

اندازه بچ (batch size) فقط یک تنظیم سیستمی (system setting) نیست، بلکه یک پارامتر بهینه‌سازی (optimization parameter) است.

---

## ۲۴. معیارهای توان عملیاتی (Throughput)

عملکرد آموزش (training performance) باید با چندین معیار (metrics) اندازه‌گیری شود.

معیارهای رایج (common metrics) شامل موارد زیر هستند:

| معیار (Metric) | معنا (Meaning) |
|---|---|
| توکن بر ثانیه (tokens per second) | کل توکن‌های پردازش‌شده (total processed tokens) در واحد زمان |
| توکن‌های مفید بر ثانیه (useful tokens per second) | توکن‌های آموزشی غیرپدینگ (non-padding training tokens) در واحد زمان |
| نمونه بر ثانیه (samples per second) | نمونه‌های پردازش‌شده (processed examples) در واحد زمان |
| زمان گام (step time) | زمان هر گام بهینه‌ساز (optimizer step) یا گام ریز-بچ (micro-batch step) |
| بهره‌برداری از FLOPs مدل (model FLOPs utilization) | توان عملیاتی به‌دست‌آمده (achieved throughput) نسبت به محاسبات نظری (theoretical compute) |
| بهره‌برداری GPU (GPU utilization) | سهم زمانی که موتورهای شتاب‌دهنده (accelerator engines) فعال هستند |
| زمان ارتباط (communication time) | زمان صرف‌شده در ارتباط توزیع‌شده (distributed communication) |
| زمان انتظار داده (data wait time) | زمان انتظار برای ورودی (input) |
| زمان نقطه بازرسی (checkpoint time) | زمانی از آموزش که صرف نقطه‌بازرسی‌گذاری (checkpointing) می‌شود |

در توکن بر ثانیه (tokens per second) باید مشخص شود که آیا توان عملیاتی به‌ازای هر دستگاه (per-device throughput) یا سراسری (global throughput) را اندازه می‌گیرد و آیا توکن‌های پدینگ (padding tokens) لحاظ شده‌اند یا نه.

یک درصد بهره‌برداری واحد (single utilization percentage) برای تشخیص عملکرد (performance diagnosis) کافی نیست.

---

## ۲۵. بهره‌وری مقیاس‌دهی (Scaling Efficiency)

بهره‌وری مقیاس‌پذیری (scaling efficiency) اندازه‌گیری می‌کند که شتاب‌دهنده‌های اضافی (additional accelerators) تا چه حد زمان اجرا (execution time) را کاهش می‌دهند یا توان عملیاتی (throughput) را افزایش می‌دهند.

برای مقیاس‌پذیری قوی (strong scaling)، بارکاری (workload) ثابت می‌ماند، در حالی که تعداد شتاب‌دهنده‌ها (accelerators) افزایش می‌یابد:

```text
strong_scaling_efficiency =
speedup / number_of_accelerators

```
برای مقیاس‌پذیری ضعیف (weak scaling)، اندازه بارکاری (workload size) با تعداد شتاب‌دهنده‌ها (accelerators) افزایش می‌یابد، در حالی که کار به‌ازای هر دستگاه (per-device work) تقریباً ثابت می‌ماند.

بهره‌وری (efficiency) معمولاً در مقیاس‌های بزرگ‌تر (larger scales) کاهش می‌یابد به دلیل:

- ارتباط بیشتر (more communication)
- سربار همگام‌سازی (synchronization overhead)
- حباب خط لوله (pipeline bubble)
- پس‌مانده‌ها (stragglers)
- رقابت شبکه (network contention)
- ابعاد ماتریسی محلی کوچک‌تر (smaller local matrix dimensions)
- سربار زمان‌بند (scheduler overhead)

یک خوشه بزرگ‌تر (larger cluster) ممکن است زودتر تمام کند (finish earlier)، در حالی که کل ساعت شتاب‌دهنده بیشتری (more total accelerator hours) مصرف می‌کند. هم زمان ساعت‌دیواری (wall-clock time) و هم هزینه کل (total cost) باید در نظر گرفته شوند.

---

## ۲۶. پروفایل‌گیری و تحلیل گلوگاه

بهینه‌سازی (optimization) باید با اندازه‌گیری (measurement) آغاز شود.

یک فرایند مفید برای پروفایل‌گیری (profiling process) به این شکل است:

```text
measure end-to-end throughput
-> identify idle periods
-> separate data, compute, and communication time
-> inspect expensive operators
-> test one change
-> verify numerical correctness
-> measure again

```

ابزار پروفایل‌گیری (profiling tool) می‌تواند موارد زیر را آشکار کند:

- خط زمانی اجرای کرنل (kernel execution timeline)
- عملیات جمعی ارتباط (communication collectives)
- انتقال میزبان به دستگاه (host-to-device transfer)
- تخصیص حافظه (memory allocation)
- سد همگام‌سازی (synchronization barrier)
- حباب خط لوله (pipeline bubble)
- شکست گراف (graph break)
- توقف بارگذار داده (data-loader stall)

پروفایل‌گیری (profiling) روی یک اجرای گرم کوتاه (short warm run) معمولاً عملی‌تر از جمع‌آوری ردیابی (trace) برای کل اجرای آموزش (entire training run) است.

---

## ۲۷. Checkpointing (ذخیرهٔ وضعیت آموزشی)

یک نقطه بازرسی (checkpoint) باید به‌اندازه کافی اطلاعات (information) داشته باشد تا آموزش را به‌درستی ازسرگیری کند (resume training correctly).

بسته به سیستم (system)، ممکن است شامل موارد زیر باشد:

- پارامترهای مدل (model parameters)
- حالت بهینه‌ساز (optimizer state)
- حالت زمان‌بند نرخ یادگیری (learning-rate scheduler state)
- حالت مقیاس‌دهنده گرادیان (gradient-scaler state)
- حالت مولد عدد تصادفی (random-number-generator state)
- موقعیت بارگذار داده (data-loader position)
- تعداد توکن مصرف‌شده (consumed-token count)
- پیکربندی آموزش (training configuration)
- نسخه توکن‌ساز و قالب (tokenizer and template version)
- فراداده توپولوژی توزیع‌شده (distributed topology metadata)

ذخیره‌کردن فقط وزن‌های مدل (model weights) برای ازسرگیری دقیق آموزش (exact training resume) کافی نیست.

طراحی نقطه بازرسی (checkpoint design) باید بین موارد زیر تعادل برقرار کند:

- بسامد ذخیره (save frequency)
- ظرفیت ذخیره‌سازی (storage capacity)
- پهنای باند نوشتن (write bandwidth)
- هدف نقطه بازیابی (recovery-point objective)
- قابلیت حمل (portability)
- نیازهای خردسازی مجدد (resharding requirements)
- هزینه اعتبارسنجی (validation cost)

یک نقطه بازرسی (checkpoint) تا زمانی که بازیابی آن (restoration) تست نشده باشد، قابل‌اعتماد (reliable) نیست.

---
## قالب‌های Checkpoint توزیع‌شده

مدل‌های توزیع‌شده (distributed models) ممکن است پارامترها (parameters) و حالت‌های بهینه‌ساز (optimizer states) را به‌صورت خرده (shards) در کارگرها (workers) ذخیره کنند.

یک نقطه بازرسی (checkpoint) ممکن است:

- در یک مدل کامل (complete model) تجمیع شود (consolidated)
- به‌صورت خرده‌های توزیع‌شده (distributed shards) ذخیره شود
- بین قالب آموزش (training format) و قالب استنتاج (inference format) تبدیل شود (converted)
- برای تعداد متفاوتی از کارگرها (workers) دوباره خرد شود (resharded)

نقطه بازرسی توزیع‌شده (distributed checkpoint) می‌تواند سریع‌تر نوشته شود (written faster) و از گردآوری کل حالت (entire state) روی یک کارگر (worker) جلوگیری کند. با این حال، ممکن است به نسخه چارچوب (framework version)، چیدمان خردسازی (sharding layout) یا اندازه جهان (world size) خاصی وابسته باشد.

یک گردش‌کار مقاوم برای نقطه بازرسی (robust checkpoint workflow) باید از موارد زیر پشتیبانی کند:

- نشانگر تکمیل اتمی (atomic completion marker)
- تشخیص خرده‌های مفقود (detection of missing shards)
- اعتبارسنجی فراداده (metadata validation)
- راستی‌آزمایی checksum
- آزمون بازیابی (restoration test)
- تبدیل به قالب انتشار قابل‌حمل (conversion to a portable release format)

نقطه بازرسی ناقص (incomplete checkpoint) هرگز نباید معتبر در نظر گرفته شود.

---

## ۲۹. تحمل خطا (Fault Tolerance)

اجراهای طولانی آموزش (long training runs) در نهایت با خرابی (failure) روبه‌رو خواهند شد.

علت‌های رایج (common causes) شامل موارد زیر هستند:

- خطای شتاب‌دهنده (accelerator error)
- خرابی میزبان (host failure)
- وقفه شبکه (network interruption)
- قطعی ذخیره‌سازی (storage outage)
- کرش فرایند (process crash)
- خطای کمبود حافظه (out-of-memory error)
- پیش‌دستی زمان‌بند (scheduler preemption)
- نقطه بازرسی خراب (corrupted checkpoint)
- ناپایداری عددی (numerical instability)

تحمل‌پذیری در برابر خطا (fault tolerance) به چیزی بیشتر از نقطه‌بازرسی‌گذاری دوره‌ای (periodic checkpointing) نیاز دارد. سیستم (system) باید خرابی را تشخیص دهد (detect failure)، کارگرهای آسیب‌دیده (affected workers) را خاتمه دهد یا ایزوله کند، یک حالت سازگار (consistent state) را بازیابی کند (restore) و بدون تکرار ناخواسته (duplicate) یا ردکردن ناخواسته داده (skip data)، ازسرگیری انجام دهد (resume).

رویه‌های بازیابی (recovery procedures) باید پیش از شروع یک اجرای پرهزینه (expensive run)، عمداً آزمون شوند (deliberately tested).

---

## ۳۰. گره‌های کند و تغییرپذیری کلاستر

آموزش هم‌زمان (synchronous training) با سرعت کندترین کارگر (slowest worker) پیش می‌رود.

یک پس‌مانده (straggler) ممکن است به علت موارد زیر باشد:

- سخت‌افزار افت‌کرده (degraded hardware)
- ازدحام شبکه (network congestion)
- رقابت بر سر ذخیره‌سازی اشتراکی (shared storage contention)
- کاهش سرعت حرارتی (thermal throttling)
- عدم‌تعادل داده (data imbalance)
- فعالیت پس‌زمینه سیستم (background system activity)
- مراحل نامتوازن خط لوله (uneven pipeline stages)
- خطاهای متناوب و تلاش مجدد (intermittent errors and retries)

پایش فقط میانگین زمان گام (average step time) می‌تواند این مسائل (issues) را پنهان کند. اغلب به زمان‌بندی به‌ازای هر رتبه (per-rank timing) و آمار ارتباط (communication statistics) نیاز است.

پس‌مانده پایدار (persistent straggler) باید بررسی یا حذف شود، چون تأخیرهای کوچک (small delays) در هزاران گام هم‌زمان (synchronized steps) انباشته می‌شوند.

---

## ۳۱. پایداری عددی

اجراهای توزیع‌شده بزرگ (large distributed runs) می‌توانند به علت ناپایداری عددی (numerical instability) شکست بخورند، حتی اگر زیرساخت سیستمی (systems infrastructure) سالم باشد.

نشانه‌های هشدار (warning signs) شامل موارد زیر هستند:

- `NaN` یا زیان بی‌نهایت (infinite loss)
- نرم گرادیان انفجاری (exploding gradient norm)
- جهش ناگهانی زیان (sudden loss spike)
- حالت بهینه‌ساز خراب (corrupted optimizer state)
- واگرایی پس از ازسرگیری (divergence after resume)
- نتیجه ناسازگار میان رتبه‌ها (inconsistent results across ranks)

محافظت‌های مفید (useful protections) شامل موارد زیر هستند:

- بریدن گرادیان (gradient clipping)
- پیاده‌سازی پایدار نرمال‌سازی (stable normalization implementation)
- مقداردهی اولیه دقیق (careful initialization)
- گرم‌کردن نرخ یادگیری (learning-rate warmup)
- کاهش آگاه از دقت (precision-aware reduction)
- تشخیص سرریز (overflow detection)
- بررسی مقدار متناهی (finite-value check)
- اعتبارسنجی پس از بازیابی نقطه بازرسی (validation after checkpoint restoration)

وقتی یک خرابی عددی (numerical failure) رخ می‌دهد، سیستم (system) باید حالت کافی (sufficient state) ثبت کند تا مشخص شود علت، داده (data)، بهینه‌سازی (optimization)، دقت (precision) یا سخت‌افزار (hardware) بوده است.

---

## ۳۲. قطعی‌بودن و بازتولیدپذیری

قطعیت دقیق (exact determinism) در آموزش توزیع‌شده روی شتاب‌دهنده (distributed accelerator training) دشوار است. بعضی کرنل‌ها (kernels)، ترتیب‌های ارتباط (communication orders) و کاهش‌های ممیز شناور (floating-point reductions) می‌توانند تفاوت‌های کوچکی بین اجراها (runs) ایجاد کنند.

بازتولیدپذیری (reproducibility) همچنان به ثبت موارد زیر نیاز دارد:

- بازبینی کد (code revision)
- نسخه‌های چارچوب و کتابخانه (framework and library versions)
- تنظیمات کامپایلر (compiler settings)
- نوع سخت‌افزار (hardware type)
- توپولوژی توزیع‌شده (distributed topology)
- بذر تصادفی (random seed)
- نسخه مجموعه‌داده (dataset version)
- ترتیب خرده‌ها (shard order)
- نسخه توکن‌ساز (tokenizer version)
- پیکربندی (configuration)
- تبار نقطه بازرسی (checkpoint lineage)

هدف همیشه خروجی یکسان در سطح بیت (bitwise-identical output) نیست. اغلب هدف این است که اطمینان حاصل شود نتایج (results) در یک بازه آماری موردانتظار (expected statistical range) باقی می‌مانند و کل تبار آموزش (training lineage) قابل بازسازی است.

---

## ۳۳. قابلیت مشاهده‌پذیری (Observability)

یک اجرای بزرگ آموزش (large training run) به پایش ساختاریافته (structured monitoring) نیاز دارد.

سیگنال‌های مهم (important signals) شامل موارد زیر هستند:

### مدل و بهینه‌سازی (Model and optimization)

- زیان آموزش (training loss)
- زیان اعتبارسنجی (validation loss)
- نرخ یادگیری (learning rate)
- نرم گرادیان (gradient norm)
- نرم پارامتر (parameter norm)
- مقیاس زیان (loss scale)
- تعداد توکن (token count)
- توزیع طول توالی (sequence-length distribution)

### سیستم‌ها (Systems)

- زمان گام (step time)
- توکن بر ثانیه (tokens per second)
- بهره‌برداری از شتاب‌دهنده (accelerator utilization)
- مصرف حافظه (memory usage)
- زمان ارتباط (communication time)
- زمان بارگذار داده (data-loader time)
- توان عملیاتی شبکه (network throughput)
- توان عملیاتی ذخیره‌سازی (storage throughput)
- دما (temperature) و توان (power)
- تعداد خطاهای سخت‌افزاری (hardware error count)

### وضعیت عملیاتی (Operational state)

- وضعیت نقطه بازرسی (checkpoint status)
- سلامت کارگر (worker health)
- راه‌اندازی مجدد کار (job restart)
- پیشرفت خرده‌داده (data-shard progress)
- تغییر پیکربندی (configuration change)
- نسخه نرم‌افزار (software version)

معیارها (metrics) باید با لاگ‌ها (logs) و ردیابی‌ها (traces) همراه باشند. معیارها نشان می‌دهند که مشکل (problem) رخ داده است؛ لاگ‌ها و ردیابی‌ها کمک می‌کنند توضیح داده شود کجا رخ داده است.

---

## ۳۴. هشداردهی (Alerting)

هشدارها (alerts) باید شرایطی (conditions) را تشخیص دهند که به مداخله انسانی (human intervention) یا مداخله خودکار (automated intervention) نیاز دارند.

نمونه‌ها (examples) شامل موارد زیر هستند:

- زیان (loss) نامتناهی می‌شود
- نرم گرادیان (gradient norm) از آستانه (threshold) عبور می‌کند
- توان عملیاتی (throughput) به‌طور معناداری افت می‌کند
- یک کارگر (worker) گزارش‌دهی را متوقف می‌کند
- ذخیره نقطه بازرسی (checkpoint save) شکست می‌خورد
- زیان اعتبارسنجی (validation loss) پسرفت می‌کند
- زمان انتظار بارگذار داده (data-loader wait time) افزایش می‌یابد
- مصرف حافظه (memory usage) به ظرفیت (capacity) نزدیک می‌شود
- خطاهای سخت‌افزاری (hardware errors) انباشته می‌شوند
- برای یک بازه تعریف‌شده (defined interval) هیچ پیشرفتی (progress) رخ نمی‌دهد

آستانه‌ها (thresholds) باید نوسان گذرا (transient variation) را از خرابی پایدار (persistent failure) متمایز کنند. هشدار پرنویز بیش‌ازحد (overly noisy alerts) باعث می‌شود رخدادهای مهم (important incidents) راحت‌تر از دست بروند.

---

## ۳۵. پیکربندی آزمایش‌ها

پیکربندی آموزش (training configuration) باید صریح (explicit)، نسخه‌بندی‌شده (versioned) و پس از شروع اجرا (run) تغییرناپذیر (immutable) باشد.

یک پیکربندی (configuration) ممکن است شامل موارد زیر باشد:

```yaml
model:
  hidden_size: 4096
  num_layers: 32
  num_attention_heads: 32

training:
  sequence_length: 4096
  global_batch_tokens: 4194304
  learning_rate: 0.0003
  precision: bf16

distributed:
  data_parallel_size: 8
  tensor_parallel_size: 4
  pipeline_parallel_size: 2
```
قالب دقیق پیکربندی (exact configuration format) کمتر از این اهمیت دارد که مقادیر نهایی حل‌شده (final resolved values) ثبت شوند.

بازنویسی با متغیر محیطی (environment-variable overrides) و فلگ‌های خط فرمان (command-line flags) نیز باید ثبت شوند. در غیر این صورت، پیکربندی ذخیره‌شده (saved configuration) ممکن است نماینده اجرایی (run) نباشد که واقعاً اجرا شده است.

---

## ۳۶. ارکستراسیون و مدیریت Jobها

ارکستراسیون خوشه (cluster orchestration)، تخصیص منبع (resource allocation)، راه‌اندازی فرایند (process startup)، پیکربندی محیط (environment configuration)، بررسی سلامت (health checks)، تلاش مجدد (retries) و پایان‌دادن به کار (job termination) را مدیریت می‌کند.

یک کار آموزش (training job) باید موارد زیر را تعریف کند:

- سخت‌افزار درخواستی (requested hardware)
- توپولوژی فرایند (process topology)
- کانتینر یا تصویر محیط (container or environment image)
- مونت ذخیره‌سازی (storage mount)
- نیاز شبکه (network requirement)
- دستور راه‌اندازی (startup command)
- سیاست خرابی (failure policy)
- محل نقطه بازرسی (checkpoint location)
- مقصد ثبت وقایع (logging destination)
- رفتار خاموش‌سازی graceful

کارگرهای توزیع‌شده (distributed workers) باید روی رتبه‌ها (ranks)، عضویت گروه (group membership) و اطلاعات rendezvous توافق داشته باشند. مقداردهی اولیه نادرست فرایند (incorrect process initialization) می‌تواند گیرکردگی (hang) ایجاد کند که شبیه خرابی شبکه (network failure) به نظر می‌رسد.

اسکریپت‌های عملیاتی (operational scripts) باید پیش از آموزش تمام‌مقیاس (full-scale training)، با یک کار چندگره‌ای کوچک (small multi-node job) تست شوند.

---

## ۳۷. امنیت و حاکمیت داده

سیستم‌های مدل زبانی بزرگ (LLM systems) اغلب داده‌های دارای مجوز (licensed data)، خصوصی (private data) یا حساس (sensitive data) را پردازش می‌کنند. بنابراین طراحی سیستم (systems design) باید کنترل دسترسی (access control) و ممیزی‌پذیری (auditability) را نیز شامل شود.

کنترل‌های مرتبط (relevant controls) شامل موارد زیر هستند:

- ذخیره‌سازی و انتقال رمزگذاری‌شده (encrypted storage and transport)
- دسترسی با کمترین سطح مجوز (least-privilege access)
- چرخش اعتبارنامه (credential rotation)
- لاگ دسترسی به مجموعه‌داده (dataset access logs)
- محیط آموزشی ایزوله (isolated training environment)
- مدیریت راز (secret management)
- سیاست نگه‌داری (retention policy)
- منشأ مصنوعات (artifact provenance)
- فراداده مجوز (license metadata)
- محدودیت دسترسی به نقطه بازرسی (checkpoint access restriction)

لاگ‌ها (logs) و مصنوعات اشکال‌زدایی (debugging artifacts) نیز می‌توانند نمونه‌های آموزشی (training examples) یا پرامپت‌ها (prompts) را افشا کنند. سامانه مشاهده‌پذیری (observability system) باید از ثبت غیرضروری محتوای حساس (sensitive content) اجتناب کند.

---

## ۳۸. مهندسی هزینه

هزینه آموزش (training cost) فقط اجاره شتاب‌دهنده (accelerator rental) نیست.

هزینه کامل (complete cost) ممکن است شامل موارد زیر باشد:

- زمان شتاب‌دهنده (accelerator time)
- منابع CPU و حافظه (CPU and memory resources)
- ذخیره‌سازی (storage)
