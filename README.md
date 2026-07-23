# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026

> 🏆 This repository is my official submission for the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Omar Mahmoud Mouhamed                                      |
| Project Name     | University AI Assistant              |
| GitHub Username  | Omar_Raghebb                                      |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en) |

---

# 📖 Project Overview

مساعد ذكي بيجاوب على أسئلة الطلاب عن اللوائح الجامعية ومتطلبات التخرج، مبني بالكامل
باستخدام تقنيات المنهج: RAG + LangChain + موديل محلي، مع Fine-tuning و Quantization
و Output Parsing و Deployment.

---

# ✨ Features

* **RAG بموديل محلي**: مافيش اعتماد على API خارجي، كل حاجة شغالة بالوزن المحلي للموديل
* **Quantization**: تقدر تقارن حجم/سرعة الموديل بـ 4-bit مقابل fp16 (`src/llm.py::benchmark()`)
* **Structured Output**: كل إجابة بترجع JSON فيه `answer` + `source_page` + `confidence`
* **Fine-tuning اختياري**: LoRA بسيط لتحسين أسلوب الرد (مش لحفظ المعلومات، ده شغل الـ RAG)
* **Deployment**: FastAPI + ngrok
* **Evaluation Harness**: مجموعة أسئلة تقييمية بتقيس نسبة الإجابات الصحيحة ومتوسط الثقة

---

# 🛠️ Technologies Used

* Python
* LangChain
* Mistral-7B (موديل محلي)
* FAISS (Vector Store)
* `pdftotext` (poppler-utils) لقراءة الـ PDF العربي
* LoRA Fine-tuning
* FastAPI + ngrok
* Streamlit

---

# ⚙️ Installation

```bash
pip install -r requirements.txt
```

> ملاحظة: الموديل والـ embeddings بيحتاجوا GPU. لو مفيش GPU على جهازك، شغّل الكود على
> Kaggle Notebook (فيه GPU مجاني T4).

---

# 🚀 Usage

1. اختبار قراءة الـ PDF:
```bash
python -m src.document_loader
```
2. بناء الـ FAISS index:
```bash
python -m src.vector_store
```
3. اختبار الموديل المحلي والـ quantization:
```bash
python -m src.llm
```
4. سؤال تجريبي من الـ RAG chain الكامل:
```bash
python -m src.chain
```
5. تشغيل التقييم:
```bash
python -m src.evaluation
```
6. (اختياري) الـ fine-tuning:
```bash
python -m src.finetune
```
7. تشغيل الواجهة محليًا:
```bash
streamlit run app.py
```
8. أو نشر السيرفر على Kaggle والاتصال بيه من جهازك:
```bash
# على Kaggle:
python deploy_server.py
# على جهازك:
python client_example.py
```

---

# 📸 Demo

_أضف صور أو GIF أو فيديو تجريبي هنا._

---

# 📈 Results

_شارك هنا نتائج التقييم (نسبة الإجابات الصحيحة، متوسط الثقة، إلخ) بعد تعديل `TEST_SET` في `src/evaluation.py`._

---

# 🔮 Future Improvements

* تحسين دقة الـ retrieval بزيادة أمثلة الـ fine-tuning
* دعم لوائح جامعية أكتر من ملف PDF واحد
* واجهة أفضل للنشر بدون الاعتماد على ngrok

---

# 📚 About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
