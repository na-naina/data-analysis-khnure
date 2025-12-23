#!/usr/bin/env python3
"""
Report Generator for Data Analysis Course
==========================================

Creates DSTU 3008:2015 compliant reports for all assignments.
Generates both DOCX and PDF formats.

Course: Intelligent Data Analysis
University: KhNURE
Teacher: Костянтин Георгійович Онищенко
"""

import os
import subprocess
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# ==================== CONFIGURATION ====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(SCRIPT_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# University info
UNIVERSITY = "МІНІСТЕРСТВО ОСВІТИ І НАУКИ УКРАЇНИ"
UNIVERSITY_NAME = "ХАРКІВСЬКИЙ НАЦІОНАЛЬНИЙ УНІВЕРСИТЕТ РАДІОЕЛЕКТРОНІКИ"
FACULTY = "Факультет комп'ютерних наук"
DEPARTMENT = "Кафедра програмної інженерії"
DISCIPLINE = "Інтелектуальний аналіз даних"
TEACHER = "ст. викл. Онищенко К.Г."
STUDENT = "студент групи МІПЗс-24-1"


# ==================== DOCUMENT HELPERS ====================

def set_cell_margins(cell, top=0, start=0, bottom=0, end=0):
    """Set cell margins."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for margin_name, margin_value in [('top', top), ('start', start), ('bottom', bottom), ('end', end)]:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), str(margin_value))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_centered_paragraph(doc, text, font_size=14, bold=False, space_after=0):
    """Add a centered paragraph."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    p.paragraph_format.space_after = Pt(space_after)
    return p


def add_paragraph(doc, text, font_size=14, bold=False, first_line_indent=1.25, space_after=0, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY):
    """Add a regular paragraph."""
    p = doc.add_paragraph()
    p.alignment = alignment
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    p.paragraph_format.first_line_indent = Cm(first_line_indent) if first_line_indent else None
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    return p


def add_heading(doc, text, level=1):
    """Add a heading."""
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        run.font.bold = True
        run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    return h


def add_image_with_caption(doc, image_path, caption, width=5.5):
    """Add an image with caption."""
    if os.path.exists(image_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(image_path, width=Inches(width))

        # Caption
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(caption)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        run.font.italic = True
        cap.paragraph_format.space_after = Pt(12)


def create_title_page(doc, work_type, work_number, work_title):
    """Create a title page according to DSTU 3008:2015."""
    # Ministry
    add_centered_paragraph(doc, UNIVERSITY, font_size=14)
    add_centered_paragraph(doc, UNIVERSITY_NAME, font_size=14, bold=True, space_after=6)
    add_centered_paragraph(doc, FACULTY, font_size=14)
    add_centered_paragraph(doc, DEPARTMENT, font_size=14, space_after=48)

    # Report title
    add_centered_paragraph(doc, "ЗВІТ", font_size=16, bold=True, space_after=6)
    add_centered_paragraph(doc, f"з {work_type} №{work_number}", font_size=14, space_after=6)
    add_centered_paragraph(doc, f'з дисципліни "{DISCIPLINE}"', font_size=14, space_after=12)
    add_centered_paragraph(doc, f'на тему: "{work_title}"', font_size=14, bold=True, space_after=72)

    # Right-aligned info block
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run(f"Виконав:\n{STUDENT}\n\nПеревірив:\n{TEACHER}")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    p.paragraph_format.space_after = Pt(120)

    # City and year
    add_centered_paragraph(doc, "Харків 2024", font_size=14)

    # Page break
    doc.add_page_break()


def add_goal_section(doc, goal_text):
    """Add goal section."""
    add_heading(doc, "1 МЕТА РОБОТИ", level=1)
    add_paragraph(doc, goal_text, first_line_indent=1.25)


def add_execution_section(doc, content_items):
    """Add execution section with content items."""
    add_heading(doc, "2 ХІД ВИКОНАННЯ РОБОТИ", level=1)

    for item in content_items:
        if item['type'] == 'text':
            add_paragraph(doc, item['content'], first_line_indent=1.25)
        elif item['type'] == 'subheading':
            add_heading(doc, item['content'], level=2)
        elif item['type'] == 'image':
            add_image_with_caption(doc, item['path'], item['caption'], item.get('width', 5.5))
        elif item['type'] == 'code':
            # Code block
            p = doc.add_paragraph()
            run = p.add_run(item['content'])
            run.font.name = 'Consolas'
            run.font.size = Pt(10)
            p.paragraph_format.space_after = Pt(6)


def add_conclusions_section(doc, conclusions):
    """Add conclusions section."""
    add_heading(doc, "3 ВИСНОВКИ", level=1)
    add_paragraph(doc, conclusions, first_line_indent=1.25)


def save_document(doc, filename):
    """Save document as DOCX and convert to PDF."""
    docx_path = os.path.join(REPORTS_DIR, f"{filename}.docx")
    pdf_path = os.path.join(REPORTS_DIR, f"{filename}.pdf")

    doc.save(docx_path)
    print(f"Saved: {docx_path}")

    # Convert to PDF using LibreOffice
    try:
        subprocess.run([
            'libreoffice', '--headless', '--convert-to', 'pdf',
            '--outdir', REPORTS_DIR, docx_path
        ], check=True, capture_output=True)
        print(f"Saved: {pdf_path}")
    except Exception as e:
        print(f"PDF conversion failed: {e}")


# ==================== REPORT GENERATORS ====================

def generate_lab1_report():
    """Generate Lab 1: Regression Analysis report."""
    doc = Document()

    # Set margins
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(1.5)

    create_title_page(doc, "лабораторної роботи", "1", "Регресійний аналіз")

    add_goal_section(doc,
        "Ознайомитися з методами регресійного аналізу даних. "
        "Вивчити побудову та аналіз моделей простої лінійної, множинної лінійної "
        "та поліноміальної регресії. Навчитися оцінювати якість регресійних моделей "
        "за допомогою коефіцієнта детермінації R², RMSE та інших метрик. "
        "Отримати практичні навички роботи з бібліотеками scikit-learn та statsmodels.")

    results_dir = os.path.join(SCRIPT_DIR, 'labs', 'lab1_regression', 'results')

    content = [
        {'type': 'subheading', 'content': '2.1 Теоретичні основи регресійного аналізу'},
        {'type': 'text', 'content':
            'Регресійний аналіз - статистичний метод дослідження впливу однієї або кількох '
            'незалежних змінних (предикторів) на залежну змінну (відгук). Метою регресії є '
            'побудова моделі, що описує цю залежність та дозволяє робити прогнози.'},
        {'type': 'text', 'content':
            'Основні метрики оцінки якості регресійної моделі:'},
        {'type': 'text', 'content':
            '• R² (коефіцієнт детермінації) - частка дисперсії залежної змінної, що пояснюється моделлю. '
            'Значення від 0 до 1, де 1 означає ідеальне пояснення.'},
        {'type': 'text', 'content':
            '• RMSE (Root Mean Square Error) - корінь із середнього квадрата помилок. '
            'Показує середню величину відхилення прогнозів від фактичних значень.'},
        {'type': 'text', 'content':
            '• MAE (Mean Absolute Error) - середня абсолютна помилка. '
            'Менш чутлива до викидів порівняно з RMSE.'},

        {'type': 'subheading', 'content': '2.2 Проста лінійна регресія'},
        {'type': 'text', 'content':
            'Проста лінійна регресія моделює залежність між однією незалежною змінною X '
            'та залежною змінною Y у вигляді лінійного рівняння:'},
        {'type': 'text', 'content': 'Y = β₀ + β₁X + ε'},
        {'type': 'text', 'content':
            'де β₀ - вільний член (intercept), що визначає значення Y при X=0; '
            'β₁ - коефіцієнт регресії (slope), що показує на скільки зміниться Y при зміні X на одиницю; '
            'ε - випадкова помилка моделі.'},
        {'type': 'text', 'content':
            'Для оцінки параметрів β₀ та β₁ використовується метод найменших квадратів (OLS), '
            'який мінімізує суму квадратів відхилень: Σ(yᵢ - ŷᵢ)² → min.'},
        {'type': 'text', 'content':
            'Для демонстрації було згенеровано вибіркові дані та побудовано модель простої '
            'лінійної регресії. На графіку показано: точки даних (синім), лінію регресії (червоним), '
            'та 95% довірчий інтервал (світло-червоним). Модель показала R² = 0.99, що свідчить '
            'про відмінну якість апроксимації.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'simple_regression.png'),
         'caption': 'Рис. 2.1 - Проста лінійна регресія з довірчим інтервалом та діагностичними графіками'},

        {'type': 'subheading', 'content': '2.3 Множинна лінійна регресія'},
        {'type': 'text', 'content':
            'Множинна лінійна регресія розширює модель простої регресії для випадку '
            'кількох незалежних змінних:'},
        {'type': 'text', 'content': 'Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε'},
        {'type': 'text', 'content':
            'Це дозволяє врахувати вплив декількох факторів на залежну змінну одночасно. '
            'Кожен коефіцієнт βᵢ інтерпретується як вплив змінної Xᵢ на Y при фіксованих '
            'значеннях інших змінних.'},
        {'type': 'text', 'content':
            'Було побудовано модель множинної регресії для прогнозування врожайності '
            'на основі факторів кількості добрив (X) та опадів (Z). '
            'Отримане рівняння: Y = 28.10 + 0.038X + 0.833Z. '
            'Модель показала R² = 0.981, Adjusted R² = 0.972.'},
        {'type': 'text', 'content':
            'Коефіцієнти моделі: збільшення добрив на 100 одиниць дає приріст врожаю на 3.8 од., '
            'збільшення опадів на 10 мм дає приріст на 8.3 од.'},

        {'type': 'subheading', 'content': '2.4 Поліноміальна регресія'},
        {'type': 'text', 'content':
            'Поліноміальна регресія використовується для моделювання нелінійних залежностей, '
            'коли зв\'язок між змінними не можна адекватно описати прямою лінією. '
            'Модель має вигляд:'},
        {'type': 'text', 'content': 'Y = β₀ + β₁X + β₂X² + ... + βₙXⁿ + ε'},
        {'type': 'text', 'content':
            'Ступінь полінома n обирається на основі аналізу даних, перевірки якості моделі '
            'та принципу парсимонії (Оккама) - не варто ускладнювати модель без потреби.'},
        {'type': 'text', 'content':
            'Для демонстрації згенеровано дані з параболічною залежністю та шумом. '
            'Побудовано поліноміальну регресію ступеня 2. На графіку видно, що крива '
            'добре апроксимує нелінійний характер даних.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'polynomial_regression.png'),
         'caption': 'Рис. 2.2 - Поліноміальна регресія ступеня 2 з візуалізацією апроксимації'},

        {'type': 'subheading', 'content': '2.5 Реалізація на Python'},
        {'type': 'text', 'content':
            'Для реалізації використано бібліотеки: scikit-learn (LinearRegression, PolynomialFeatures), '
            'statsmodels (OLS з детальною статистикою), numpy (числові операції), '
            'matplotlib та seaborn (візуалізація).'},
    ]

    add_execution_section(doc, content)

    add_conclusions_section(doc,
        "У ході виконання лабораторної роботи було досліджено методи регресійного аналізу: "
        "просту лінійну, множинну лінійну та поліноміальну регресію. "
        "Вивчено теоретичні основи методу найменших квадратів та метрики оцінки якості моделей. "
        "Реалізовано алгоритми побудови регресійних моделей мовою Python з використанням "
        "бібліотек scikit-learn та statsmodels. "
        "Проведено оцінку якості моделей за метриками R², RMSE, MAE. "
        "Візуалізовано результати регресійного аналізу з довірчими інтервалами та діагностичними графіками. "
        "Отримані знання можуть бути застосовані для прогнозування та аналізу залежностей у реальних даних.")

    save_document(doc, 'lab1_regression')


def generate_lab2_report():
    """Generate Lab 2: Clustering and Decision Trees report."""
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(1.5)

    create_title_page(doc, "лабораторної роботи", "2", "Кластеризація та дерева рішень")

    add_goal_section(doc,
        "Ознайомитися з методами кластеризації даних та класифікації за допомогою дерев рішень. "
        "Вивчити алгоритми K-Means, ієрархічної кластеризації, DBSCAN. "
        "Навчитися будувати та інтерпретувати дерева рішень для задач класифікації.")

    results_dir = os.path.join(SCRIPT_DIR, 'labs', 'lab2_clustering', 'results')

    content = [
        {'type': 'subheading', 'content': '2.1 Визначення оптимальної кількості кластерів'},
        {'type': 'text', 'content':
            'Для визначення оптимальної кількості кластерів використано метод ліктя (Elbow method) '
            'та коефіцієнт силуету (Silhouette score). Метод ліктя базується на аналізі інерції '
            '(сума квадратів відстаней до центроїдів), а силует оцінює якість кластеризації.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'elbow_silhouette.png'),
         'caption': 'Рис. 2.1 - Метод ліктя та коефіцієнт силуету'},

        {'type': 'subheading', 'content': '2.2 Кластеризація K-Means'},
        {'type': 'text', 'content':
            'Алгоритм K-Means ітеративно призначає точки до найближчих центроїдів та '
            'оновлює положення центроїдів. Результати кластеризації для k=4 показані на графіку.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'kmeans_clusters.png'),
         'caption': 'Рис. 2.2 - Результати кластеризації K-Means (k=4)'},

        {'type': 'subheading', 'content': '2.3 Ієрархічна кластеризація'},
        {'type': 'text', 'content':
            'Ієрархічна кластеризація будує дерево (дендрограму) об\'єднання або поділу кластерів. '
            'Використано метод Уорда (Ward linkage), який мінімізує дисперсію всередині кластерів.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'dendrogram.png'),
         'caption': 'Рис. 2.3 - Дендрограма ієрархічної кластеризації'},
        {'type': 'image', 'path': os.path.join(results_dir, 'hierarchical_clusters.png'),
         'caption': 'Рис. 2.4 - Результати ієрархічної кластеризації'},

        {'type': 'subheading', 'content': '2.4 Кластеризація DBSCAN'},
        {'type': 'text', 'content':
            'DBSCAN (Density-Based Spatial Clustering of Applications with Noise) - '
            'алгоритм кластеризації на основі щільності. Він автоматично визначає кількість '
            'кластерів та ідентифікує шумові точки (викиди).'},
        {'type': 'image', 'path': os.path.join(results_dir, 'dbscan_clusters.png'),
         'caption': 'Рис. 2.5 - Результати кластеризації DBSCAN'},

        {'type': 'subheading', 'content': '2.5 Дерево рішень'},
        {'type': 'text', 'content':
            'Дерево рішень - модель класифікації, що розбиває простір ознак на регіони '
            'за допомогою послідовних бінарних розбиттів. Побудовано дерево для класифікації '
            'набору даних Iris.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'decision_tree.png'),
         'caption': 'Рис. 2.6 - Структура дерева рішень', 'width': 6.5},
        {'type': 'image', 'path': os.path.join(results_dir, 'confusion_matrix.png'),
         'caption': 'Рис. 2.7 - Матриця помилок класифікації'},
        {'type': 'image', 'path': os.path.join(results_dir, 'feature_importance.png'),
         'caption': 'Рис. 2.8 - Важливість ознак у дереві рішень'},
    ]

    add_execution_section(doc, content)

    add_conclusions_section(doc,
        "У ході виконання лабораторної роботи було досліджено методи кластеризації "
        "(K-Means, ієрархічна, DBSCAN) та класифікації (дерева рішень). "
        "Реалізовано алгоритми визначення оптимальної кількості кластерів. "
        "Проведено порівняльний аналіз різних методів кластеризації. "
        "Побудовано та візуалізовано дерево рішень для задачі класифікації. "
        "Оцінено якість класифікації за допомогою матриці помилок та метрик accuracy, precision, recall.")

    save_document(doc, 'lab2_clustering')


def generate_lab3_report():
    """Generate Lab 3: Apriori Algorithm report."""
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(1.5)

    create_title_page(doc, "лабораторної роботи", "3", "Алгоритм Apriori та асоціативні правила")

    add_goal_section(doc,
        "Ознайомитися з алгоритмом Apriori для пошуку частих наборів елементів та "
        "генерації асоціативних правил. Вивчити метрики support, confidence, lift. "
        "Застосувати алгоритм до аналізу транзакційних даних (market basket analysis).")

    results_dir = os.path.join(SCRIPT_DIR, 'labs', 'lab3_apriori', 'results')

    content = [
        {'type': 'subheading', 'content': '2.1 Теоретичні основи'},
        {'type': 'text', 'content':
            'Алгоритм Apriori використовує властивість анти-монотонності: якщо набір елементів '
            'не є частим, то жоден його надмножина також не є частою. Це дозволяє ефективно '
            'відсікати простір пошуку.'},
        {'type': 'text', 'content':
            'Основні метрики: Support - частка транзакцій з даним набором; '
            'Confidence - умовна ймовірність правила; Lift - відношення confidence до очікуваної.'},

        {'type': 'subheading', 'content': '2.2 Розподіл підтримки'},
        {'type': 'image', 'path': os.path.join(results_dir, 'support_distribution.png'),
         'caption': 'Рис. 2.1 - Розподіл підтримки частих наборів'},

        {'type': 'subheading', 'content': '2.3 Метрики асоціативних правил'},
        {'type': 'image', 'path': os.path.join(results_dir, 'rules_metrics.png'),
         'caption': 'Рис. 2.2 - Метрики знайдених асоціативних правил'},

        {'type': 'subheading', 'content': '2.4 Теплова карта правил'},
        {'type': 'image', 'path': os.path.join(results_dir, 'rules_heatmap.png'),
         'caption': 'Рис. 2.3 - Теплова карта асоціативних правил'},

        {'type': 'subheading', 'content': '2.5 Мережа асоціацій'},
        {'type': 'image', 'path': os.path.join(results_dir, 'rules_network.png'),
         'caption': 'Рис. 2.4 - Мережева візуалізація асоціативних правил'},

        {'type': 'subheading', 'content': '2.6 Матриця частот'},
        {'type': 'image', 'path': os.path.join(results_dir, 'frequency_matrix.png'),
         'caption': 'Рис. 2.5 - Матриця спільної появи товарів'},

        {'type': 'subheading', 'content': '2.7 Аналіз чутливості'},
        {'type': 'text', 'content':
            'Проведено аналіз чутливості кількості знайдених правил до порогових значень '
            'min_support та min_confidence.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'sensitivity_analysis.png'),
         'caption': 'Рис. 2.6 - Аналіз чутливості до порогових значень'},
    ]

    add_execution_section(doc, content)

    add_conclusions_section(doc,
        "У ході виконання лабораторної роботи було реалізовано алгоритм Apriori для пошуку "
        "частих наборів елементів та генерації асоціативних правил. "
        "Проаналізовано транзакційні дані супермаркету та виявлено закономірності у покупках. "
        "Візуалізовано результати у вигляді теплових карт, мережевих графів та матриць частот. "
        "Досліджено вплив порогових значень support та confidence на кількість знайдених правил. "
        "Отримані знання застосовуються у рекомендаційних системах та маркетинговому аналізі.")

    save_document(doc, 'lab3_apriori')


def generate_lab4_report():
    """Generate Lab 4: Genetic Algorithms report."""
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(1.5)

    create_title_page(doc, "лабораторної роботи", "4", "Генетичні алгоритми")

    add_goal_section(doc,
        "Ознайомитися з генетичними алгоритмами як методом евристичної оптимізації. "
        "Вивчити основні оператори: селекцію, кросовер, мутацію. "
        "Застосувати генетичний алгоритм до оптимізації тестових функцій.")

    results_dir = os.path.join(SCRIPT_DIR, 'labs', 'lab4_genetic_algorithms', 'results')

    content = [
        {'type': 'subheading', 'content': '2.1 Теоретичні основи'},
        {'type': 'text', 'content':
            'Генетичний алгоритм - метаевристичний метод оптимізації, що імітує процес '
            'природного відбору. Популяція рішень еволюціонує через оператори селекції, '
            'кросоверу та мутації.'},

        {'type': 'subheading', 'content': '2.2 Оптимізація функції Sphere'},
        {'type': 'text', 'content':
            'Функція Sphere f(x) = Σxᵢ² - проста унімодальна функція з глобальним мінімумом '
            'у точці (0, 0, ..., 0).'},
        {'type': 'image', 'path': os.path.join(results_dir, 'sphere_evolution.png'),
         'caption': 'Рис. 2.1 - Еволюція оптимізації функції Sphere'},
        {'type': 'image', 'path': os.path.join(results_dir, 'sphere_surface.png'),
         'caption': 'Рис. 2.2 - Поверхня функції Sphere та знайдений оптимум', 'width': 6.5},

        {'type': 'subheading', 'content': '2.3 Оптимізація функції Rastrigin'},
        {'type': 'text', 'content':
            'Функція Rastrigin - складна мультимодальна функція з багатьма локальними мінімумами. '
            'Це серйозний тест для оптимізаційних алгоритмів.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'rastrigin_evolution.png'),
         'caption': 'Рис. 2.3 - Еволюція оптимізації функції Rastrigin'},
        {'type': 'image', 'path': os.path.join(results_dir, 'rastrigin_surface.png'),
         'caption': 'Рис. 2.4 - Поверхня функції Rastrigin', 'width': 6.5},

        {'type': 'subheading', 'content': '2.4 Оптимізація функції Rosenbrock'},
        {'type': 'text', 'content':
            'Функція Rosenbrock (banana function) має вузьку параболічну долину, '
            'що ускладнює пошук глобального мінімуму.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'rosenbrock_evolution.png'),
         'caption': 'Рис. 2.5 - Еволюція оптимізації функції Rosenbrock'},
        {'type': 'image', 'path': os.path.join(results_dir, 'rosenbrock_surface.png'),
         'caption': 'Рис. 2.6 - Поверхня функції Rosenbrock', 'width': 6.5},

        {'type': 'subheading', 'content': '2.5 Порівняння методів селекції'},
        {'type': 'text', 'content':
            'Порівняно три методи селекції: турнірна, рулеткова та рангова. '
            'Експеримент проведено на функції Sphere.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'selection_comparison.png'),
         'caption': 'Рис. 2.7 - Порівняння методів селекції'},

        {'type': 'subheading', 'content': '2.6 Багатовимірна оптимізація'},
        {'type': 'image', 'path': os.path.join(results_dir, 'sphere_10d_evolution.png'),
         'caption': 'Рис. 2.8 - Оптимізація 10-вимірної функції Sphere'},
    ]

    add_execution_section(doc, content)

    add_conclusions_section(doc,
        "У ході виконання лабораторної роботи було реалізовано генетичний алгоритм "
        "для задач неперервної оптимізації. Досліджено оптимізацію тестових функцій "
        "Sphere, Rastrigin та Rosenbrock. Порівняно ефективність різних методів селекції: "
        "турнірної, рулеткової та рангової. Проведено експерименти з багатовимірною оптимізацією. "
        "Генетичні алгоритми показали здатність знаходити глобальні оптимуми складних функцій.")

    save_document(doc, 'lab4_genetic_algorithms')


def generate_practical1_report():
    """Generate Practical 1: OLS Regression report."""
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(1.5)

    create_title_page(doc, "практичної роботи", "1", "Метод найменших квадратів (OLS)")

    add_goal_section(doc,
        "Вивчити метод найменших квадратів (Ordinary Least Squares, OLS) для побудови "
        "регресійних моделей. Провести аналіз реальних економічних даних підприємства. "
        "Оцінити якість моделі та перевірити виконання припущень OLS.")

    results_dir = os.path.join(SCRIPT_DIR, 'practicals', 'practical1_ols', 'results')

    content = [
        {'type': 'subheading', 'content': '2.1 Опис даних'},
        {'type': 'text', 'content':
            'Використано дані виробничо-економічної діяльності машинобудівного підприємства. '
            'Залежна змінна Y1 - продуктивність праці. Незалежні змінні X1-X10 характеризують '
            'різні аспекти діяльності підприємства.'},

        {'type': 'subheading', 'content': '2.2 Кореляційний аналіз'},
        {'type': 'text', 'content':
            'Побудовано матрицю кореляцій для визначення зв\'язків між змінними та '
            'виявлення потенційної мультиколінеарності.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'correlation_matrix.png'),
         'caption': 'Рис. 2.1 - Матриця кореляцій змінних'},

        {'type': 'subheading', 'content': '2.3 Побудова OLS моделі'},
        {'type': 'text', 'content':
            'Побудовано регресійну модель методом найменших квадратів. '
            'Проведено оцінку значущості коефіцієнтів та загальної якості моделі.'},

        {'type': 'subheading', 'content': '2.4 Діагностика моделі'},
        {'type': 'text', 'content':
            'Проведено діагностику моделі: аналіз залишків, перевірка нормальності розподілу, '
            'виявлення впливових спостережень.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'regression_diagnostics.png'),
         'caption': 'Рис. 2.2 - Діагностичні графіки OLS регресії', 'width': 6.5},
    ]

    add_execution_section(doc, content)

    add_conclusions_section(doc,
        "У ході виконання практичної роботи було застосовано метод найменших квадратів "
        "для аналізу економічних даних підприємства. Проведено кореляційний аналіз та "
        "відібрано значущі предиктори. Побудовано та оцінено OLS регресійну модель. "
        "Виконано діагностику моделі та перевірено виконання припущень методу. "
        "Результати можуть бути використані для прогнозування продуктивності праці.")

    save_document(doc, 'practical1_ols')


def generate_practical2_report():
    """Generate Practical 2: TF-IDF report."""
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(1.5)

    create_title_page(doc, "практичної роботи", "2", "Класифікація тексту методом TF-IDF")

    add_goal_section(doc,
        "Вивчити методи представлення тексту у вигляді числових векторів. "
        "Реалізувати алгоритм TF-IDF (Term Frequency - Inverse Document Frequency). "
        "Застосувати косинусну подібність для ранжування документів за запитом.")

    results_dir = os.path.join(SCRIPT_DIR, 'practicals', 'practical2_tfidf', 'results')

    content = [
        {'type': 'subheading', 'content': '2.1 Теоретичні основи TF-IDF'},
        {'type': 'text', 'content':
            'TF-IDF - метод оцінки важливості слова в документі відносно колекції. '
            'TF (Term Frequency) - частота слова в документі. '
            'IDF (Inverse Document Frequency) - обернена частота документів зі словом. '
            'Вага слова: w = TF × log(N/DF), де N - кількість документів.'},

        {'type': 'subheading', 'content': '2.2 Задача (Варіант 1)'},
        {'type': 'text', 'content':
            'Дано запит: "silver black car" та 4 документи. '
            'Необхідно обчислити TF-IDF ваги та ранжувати документи за релевантністю '
            'використовуючи косинусну подібність.'},

        {'type': 'subheading', 'content': '2.3 Результати ранжування'},
        {'type': 'image', 'path': os.path.join(results_dir, 'similarity_scores.png'),
         'caption': 'Рис. 2.1 - Косинусна подібність документів до запиту'},

        {'type': 'subheading', 'content': '2.4 Матриця TF-IDF ваг'},
        {'type': 'image', 'path': os.path.join(results_dir, 'tfidf_heatmap.png'),
         'caption': 'Рис. 2.2 - Теплова карта TF-IDF ваг', 'width': 6.5},
    ]

    add_execution_section(doc, content)

    add_conclusions_section(doc,
        "У ході виконання практичної роботи було реалізовано алгоритм TF-IDF "
        "для представлення текстових документів у вигляді числових векторів. "
        "Обчислено косинусну подібність між запитом та документами. "
        "Проведено ранжування документів за релевантністю. "
        "Документ D1 виявився найбільш релевантним запиту з similarity = 0.695. "
        "TF-IDF широко застосовується в пошукових системах та інформаційному пошуку.")

    save_document(doc, 'practical2_tfidf')


def generate_research_report():
    """Generate Research: RNN vs LSTM report."""
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(1.5)

    create_title_page(doc, "індивідуального завдання (ІНДЗ)", "",
                      "Порівняння RNN та LSTM для обробки природної мови")

    add_goal_section(doc,
        "Провести порівняльний аналіз рекурентних нейронних мереж (RNN) та мереж "
        "з довгою короткостроковою пам'яттю (LSTM) для задач обробки послідовностей. "
        "Дослідити проблему зникаючого градієнта та переваги архітектури LSTM.")

    results_dir = os.path.join(SCRIPT_DIR, 'research', 'results')

    content = [
        {'type': 'subheading', 'content': '2.1 Архітектура RNN'},
        {'type': 'text', 'content':
            'Рекурентна нейронна мережа (RNN) обробляє послідовності, передаючи прихований стан '
            'з кожного часового кроку на наступний. Формула оновлення: h(t) = tanh(W·h(t-1) + U·x(t) + b). '
            'RNN страждає від проблеми зникаючого градієнта при довгих послідовностях.'},

        {'type': 'subheading', 'content': '2.2 Архітектура LSTM'},
        {'type': 'text', 'content':
            'LSTM (Long Short-Term Memory) вирішує проблему зникаючого градієнта через механізм '
            'воріт: forget gate, input gate, output gate. Додатковий cell state дозволяє '
            'зберігати інформацію на довгих часових проміжках.'},

        {'type': 'subheading', 'content': '2.3 Експеримент 1: Стандартні послідовності'},
        {'type': 'text', 'content':
            'Порівняння на задачі класифікації послідовностей середньої довжини (50 кроків). '
            'Обидві архітектури показують схожі результати на простих задачах.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'exp1_training.png'),
         'caption': 'Рис. 2.1 - Криві навчання на стандартних послідовностях', 'width': 6.5},

        {'type': 'subheading', 'content': '2.4 Експеримент 2: Довгострокові залежності'},
        {'type': 'text', 'content':
            'Тестування здатності моделей запам\'ятовувати інформацію з початку послідовності. '
            'LSTM значно перевершує RNN на цій задачі.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'exp2_training.png'),
         'caption': 'Рис. 2.2 - Криві навчання на довгих залежностях', 'width': 6.5},

        {'type': 'subheading', 'content': '2.5 Експеримент 3: Вплив довжини послідовності'},
        {'type': 'text', 'content':
            'Дослідження деградації точності при збільшенні довжини послідовності. '
            'RNN швидко втрачає здатність навчатися на довгих послідовностях.'},
        {'type': 'image', 'path': os.path.join(results_dir, 'exp3_seq_length.png'),
         'caption': 'Рис. 2.3 - Залежність точності від довжини послідовності'},
    ]

    add_execution_section(doc, content)

    add_conclusions_section(doc,
        "Проведено експериментальне порівняння архітектур RNN та LSTM. "
        "На стандартних послідовностях обидві моделі показують схожу точність. "
        "LSTM значно перевершує RNN на задачах з довгостроковими залежностями. "
        "При збільшенні довжини послідовності точність RNN швидко падає, тоді як LSTM "
        "зберігає здатність навчатися. Cell state в LSTM діє як \"шосе\" для градієнтів, "
        "дозволяючи інформації зберігатися на багатьох часових кроках. "
        "Рекомендується використовувати LSTM для задач NLP з довгими текстами.")

    save_document(doc, 'research_rnn_lstm')


# ==================== MAIN ====================

def main():
    """Generate all reports."""
    print("=" * 60)
    print("Report Generator for Data Analysis Course")
    print("=" * 60)

    print("\nGenerating Lab 1: Regression Analysis...")
    generate_lab1_report()

    print("\nGenerating Lab 2: Clustering and Decision Trees...")
    generate_lab2_report()

    print("\nGenerating Lab 3: Apriori Algorithm...")
    generate_lab3_report()

    print("\nGenerating Lab 4: Genetic Algorithms...")
    generate_lab4_report()

    print("\nGenerating Practical 1: OLS Regression...")
    generate_practical1_report()

    print("\nGenerating Practical 2: TF-IDF...")
    generate_practical2_report()

    print("\nGenerating Research: RNN vs LSTM...")
    generate_research_report()

    print("\n" + "=" * 60)
    print(f"All reports saved to: {REPORTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
