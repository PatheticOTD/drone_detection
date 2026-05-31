#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove redundant English glosses + format formulas per GOST (OMML, per-section numbering).
Edits word/document.xml directly. Refuses to run while the file is locked."""
import zipfile, shutil, os, sys
from pathlib import Path

FN = "Обнаружение_и_распознавание_дронов_последняя_версия.docx"
BACKUP = FN + ".before-gost"
LOCK = f".~lock.{FN}#"
if Path(LOCK).exists() and os.environ.get("FORCE") != "1":
    sys.exit(f"ABORT: файл открыт в редакторе ({LOCK}). Закройте документ и запустите снова.")

def esc(t):
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

# ---- paragraph builders (Liberation Serif, как в документе) ----
WRPR = ('<w:rFonts w:ascii="Liberation Serif" w:hAnsi="Liberation Serif" '
        'w:eastAsia="Liberation Serif" w:cs="Liberation Serif" />'
        '<w:color w:val="000000" w:themeColor="text1" /><w:sz w:val="28" /><w:szCs w:val="28" />')
PPR_BODY = ('<w:keepLines w:val="true" /><w:pBdr></w:pBdr><w:spacing w:line="360" w:lineRule="auto" />'
            '<w:ind w:firstLine="720" />')

def trun(text):
    return f'<w:r><w:rPr>{WRPR}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'

def body(text):
    return f'<w:p><w:pPr>{PPR_BODY}<w:jc w:val="both" /><w:rPr>{WRPR}</w:rPr></w:pPr>{trun(text)}</w:p>'

def para_mixed(parts):
    """parts: list of str (text) or ('m', omml_inner) (inline math)."""
    out=[f'<w:p><w:pPr>{PPR_BODY}<w:jc w:val="both" /><w:rPr>{WRPR}</w:rPr></w:pPr>']
    for p in parts:
        if isinstance(p, tuple) and p[0]=="m": out.append(f'<m:oMath>{p[1]}</m:oMath>')
        else: out.append(trun(p))
    out.append('</w:p>')
    return "".join(out)

def formula_para(inner, num):
    tabs = '<w:tabs><w:tab w:val="center" w:pos="4819"/><w:tab w:val="right" w:pos="9638"/></w:tabs>'
    return (f'<w:p><w:pPr>{tabs}<w:spacing w:line="360" w:lineRule="auto"/><w:jc w:val="left"/>'
            f'<w:rPr>{WRPR}</w:rPr></w:pPr>'
            f'<w:r><w:rPr>{WRPR}</w:rPr><w:tab/></w:r>'
            f'<m:oMath>{inner}</m:oMath>'
            f'<w:r><w:rPr>{WRPR}</w:rPr><w:tab/><w:t xml:space="preserve">({num})</w:t></w:r></w:p>')

# ---- OMML builders (Cambria Math, как у существующих «м²») ----
MR = ('<w:rPr><w:rFonts w:hint="default" w:ascii="Cambria Math" w:hAnsi="Cambria Math" '
      'w:eastAsia="Cambria Math" w:cs="Cambria Math" /><w:color w:val="000000" w:themeColor="text1" />'
      '<w:lang w:val="ru-RU" /></w:rPr>')
CTRL = ('<m:ctrlPr><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" '
        'w:eastAsia="Cambria Math" w:cs="Cambria Math" /><w:color w:val="000000" w:themeColor="text1" />'
        '</w:rPr></m:ctrlPr>')
def mr(text, nor=False):
    mpr = '<m:rPr><m:nor/></m:rPr>' if nor else '<m:rPr></m:rPr>'
    return f'<m:r>{MR}{mpr}<m:t>{esc(text)}</m:t></m:r>'
def frac(num, den):
    return f'<m:f><m:fPr>{CTRL}</m:fPr><m:num>{num}</m:num><m:den>{den}</m:den></m:f>'
def ssub(b, s):
    return f'<m:sSub><m:sSubPr>{CTRL}</m:sSubPr><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>'
def ssup(b, s):
    return f'<m:sSup><m:sSupPr>{CTRL}</m:sSupPr><m:e>{b}</m:e><m:sup>{s}</m:sup></m:sSup>'
def ssubsup(b, sub, sup):
    return (f'<m:sSubSup><m:sSubSupPr>{CTRL}</m:sSubSupPr><m:e>{b}</m:e>'
            f'<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>')
def rad(e):
    return (f'<m:rad><m:radPr><m:degHide m:val="1"/>{CTRL}</m:radPr><m:deg></m:deg>'
            f'<m:e>{e}</m:e></m:rad>')
def nary(sub, e):
    return (f'<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/><m:supHide m:val="1"/>'
            f'{CTRL}</m:naryPr><m:sub>{sub}</m:sub><m:sup></m:sup><m:e>{e}</m:e></m:nary>')

# variable fragments
v_i=mr("i")
C_sv=ssub(mr("C"), mr("св",nor=True))
w_i=ssub(mr("w"), mr("i")); c_i=ssub(mr("c"), mr("i"))
sig_i=ssub(mr("σ"), mr("i")); sig_i2=ssubsup(mr("σ"), mr("i"), mr("2"))
sig_sv=ssub(mr("σ"), mr("св",nor=True))
F1=ssub(mr("F"), mr("1"))

# ---- formula OMML ----
F_21 = mr("R")+mr("≈")+frac(mr("D"), mr("θ"))
F_41 = C_sv+mr("=")+frac(nary(v_i, w_i+c_i), nary(v_i, w_i))
F_42 = w_i+mr("=")+frac(c_i, sig_i2)
F_43 = sig_sv+mr("=")+frac(mr("1"), rad(nary(v_i, w_i)))
F_51 = mr("P")+mr("=")+frac(mr("ИП",nor=True), mr("ИП",nor=True)+mr("+")+mr("ЛП",nor=True))
F_52 = mr("R")+mr("=")+frac(mr("ИП",nor=True), mr("ИП",nor=True)+mr("+")+mr("ЛО",nor=True))
F_53 = F1+mr("=")+frac(mr("2")+mr("P")+mr("R"), mr("P")+mr("+")+mr("R"))

# ---- load ----
with zipfile.ZipFile(FN) as zin:
    members=[(it, zin.read(it.filename)) for it in zin.infolist()]
xml=next(d for it,d in members if it.filename=="word/document.xml").decode("utf-8")

def replace_para(anchor, new_xml):
    i=xml_ref[0].find(anchor); assert i!=-1, f"anchor not found: {anchor!r}"
    s=max(xml_ref[0].rfind("<w:p>",0,i), xml_ref[0].rfind("<w:p ",0,i))
    e=xml_ref[0].find("</w:p>",i)+len("</w:p>")
    xml_ref[0]=xml_ref[0][:s]+new_xml+xml_ref[0][e:]
xml_ref=[xml]

# ===== Part 2: formula paragraphs =====
# §2.4
replace_para("R ≈ D / θ",
  body("Дальность оценивается методом угловой триангуляции при известном характерном линейном "
       "размере дрона по формуле (2.1):")
  + formula_para(F_21, "2.1")
  + para_mixed(["где ", ("m",mr("R")), " — дальность до цели, м; ", ("m",mr("D")),
       " — характерный линейный размер дрона, м; ", ("m",mr("θ")),
       " — угловой размер ограничивающей рамки, рад."])
  + body("Погрешность составляет ±20% при известном типе дрона и достигает 50% при неизвестном "
       "размере. Направление определяется непосредственно по координатам ограничивающей рамки в "
       "кадре и параметрам наведения камеры. Угловая точность составляет 0,1–0,5° в зависимости "
       "от разрешения матрицы."))
# §4.5 fusion confidence
replace_para("C_св = Σ (w_i",
  body("Сводная оценка достоверности вычисляется как взвешенное среднее по доступным каналам по "
       "формуле (4.1):")
  + formula_para(F_41, "4.1")
  + para_mixed(["где ", ("m",C_sv), " — сводная оценка достоверности; ", ("m",w_i),
       " — вес i-го канала из параметра SENSOR_WEIGHTS; ", ("m",c_i),
       " — достоверность, сообщённая i-м каналом; суммирование ведётся по доступным каналам."])
  + body("Веса можно изменять во время работы методом update_weights() с проверкой суммы, равной "
       "единице, что позволяет менять важность каналов без перезапуска системы — например, "
       "увеличивать вес радара ночью и снижать вес видео."))
# §4.5 coords (two formulas)
replace_para("1 / √(Σ w_i)",
  body("Объединение координат выполняется методом обратного взвешивания по дисперсии: каждой "
       "оценке местоположения присваивается вес, равный отношению достоверности канала к квадрату "
       "его пространственной погрешности, по формуле (4.2):")
  + formula_para(F_42, "4.2")
  + para_mixed(["где ", ("m",w_i), " — вес оценки местоположения i-го канала; ", ("m",c_i),
       " — достоверность i-го канала; ", ("m",sig_i),
       " — пространственная погрешность i-го канала, м."])
  + body("В объединении участвуют только каналы с известным азимутом (радар и видео); акустический "
       "и радиочастотный каналы дают лишь оценку дальности и сохраняются в показаниях отдельно. "
       "Итоговая среднеквадратичная погрешность определения координат вычисляется по формуле (4.3):")
  + formula_para(F_43, "4.3")
  + para_mixed(["где ", ("m",sig_sv), " — итоговая среднеквадратичная погрешность определения "
       "координат, м; ", ("m",w_i), " — весовые коэффициенты по формуле (4.2)."])
  + body("Погрешность уменьшается при добавлении каждого нового канала."))
# §5.4.1 metrics
replace_para("F1 = 2·P·R",
  body("Для классификаторов RadarResNet, AudioResNet и RF-ResNet-1D используются стандартные "
       "метрики многоклассовой классификации. Точность — доля верно классифицированных образцов "
       "от общего числа. Точность по классу вычисляется по формуле (5.1), полнота — по формуле "
       "(5.2), а их гармоническое среднее (F1-мера) — по формуле (5.3):")
  + formula_para(F_51, "5.1") + formula_para(F_52, "5.2") + formula_para(F_53, "5.3")
  + para_mixed(["где ", ("m",mr("P")), " — точность по классу; ", ("m",mr("R")),
       " — полнота по классу; ИП, ЛП, ЛО — число истинно положительных, ложно положительных и "
       "ложно отрицательных решений соответственно."])
  + body("Для многоклассовых задач приводится макро-F1 — простое среднее F1-меры по всем классам, "
       "не учитывающее их размер. Дополнительно строится матрица ошибок, по строкам которой "
       "откладывается истинный класс, по столбцам — предсказанный."))

xml=xml_ref[0]

# ===== Part 1: remove redundant English glosses (вне перестроенных абзацев) =====
def rep1(old, new):
    global xml
    assert xml.count(old)==1, f"count={xml.count(old)} for {old!r}"
    xml=xml.replace(old, new)
rep1("(dropout 0,5)", "0,5")
for g in [" (max-pooling)", " (residual)", " (baseline)", " (fine-tuning)", " (backbone)",
          " (neck)", " (NMS)", " (mixup)", " (Real-Time Detection Transformer)"]:
    rep1(g, "")
rep1("«cosine annealing»", "«косинусное затухание»")
rep1("mean Average Precision, ", "")
rep1("анкер-фри (anchor-free)", "безъякорный")
rep1("анкер-фри", "безъякорный")   # second occurrence

# ---- backup + rewrite ----
shutil.copy2(FN, BACKUP)
with zipfile.ZipFile(FN, "w", zipfile.ZIP_DEFLATED) as zout:
    for it, data in members:
        if it.filename=="word/document.xml": data=xml.encode("utf-8")
        zout.writestr(it, data)
print("OK -> backup:", BACKUP)
