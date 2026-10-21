"""Regenerate the flash fixtures: python tests/data/flash/make_fixtures.py

Needs pymupdf and pymupdf-fonts (FiraGO), neither a PageIndex dependency. Each
PDF is a title page and three 20pt headings over 11pt body lines, with the font
subset embedded. Output is byte-stable, so a regeneration leaves git clean.
"""
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent

ZH = ["本公司致力于为客户提供高质量的产品和服务，持续推动技术创新与业务增长。",
      "报告期内，公司实现营业收入同比增长，主要得益于核心业务的稳步扩张。",
      "管理层将继续优化资源配置，加强风险管理，提升整体运营效率。",
      "未来公司将围绕战略目标，深化数字化转型，拓展新的市场机会。",
      "董事会对全体员工的辛勤付出表示衷心感谢，并对未来发展充满信心。"]
JA = ["当社は、お客様に高品質な製品とサービスを提供し、技術革新と事業成長を推進しています。",
      "当期において、当社の売上高は主力事業の着実な拡大により前年同期比で増加しました。",
      "経営陣は引き続き資源配分を最適化し、リスク管理を強化して業務効率を高めていきます。",
      "今後は戦略目標を軸にデジタル変革を深め、新たな市場機会の開拓を進めてまいります。",
      "取締役会は全従業員の努力に心より感謝し、今後の発展に自信を持っております。"]
HI = ["कंपनी ग्राहकों को उच्च गुणवत्ता वाले उत्पाद और सेवाएँ प्रदान करने के लिए प्रतिबद्ध है।",
      "रिपोर्टिंग अवधि में कंपनी की आय में मुख्य व्यवसाय के विस्तार के कारण वृद्धि हुई।",
      "प्रबंधन संसाधनों का अनुकूलन और जोखिम प्रबंधन को मजबूत करना जारी रखेगा।",
      "भविष्य में कंपनी रणनीतिक लक्ष्यों के अनुरूप डिजिटल परिवर्तन को गहरा करेगी।",
      "निदेशक मंडल सभी कर्मचारियों की कड़ी मेहनत के लिए हृदय से आभार व्यक्त करता है।"]
AR = ["تلتزم الشركة بتقديم منتجات وخدمات عالية الجودة لعملائها في جميع الأسواق.",
      "خلال فترة التقرير ارتفعت إيرادات الشركة بفضل التوسع المستمر في الأعمال الأساسية.",
      "ستواصل الإدارة تحسين توزيع الموارد وتعزيز إدارة المخاطر لرفع الكفاءة التشغيلية.",
      "في المستقبل ستعمق الشركة التحول الرقمي وفق أهدافها الاستراتيجية وتستكشف أسواقا جديدة.",
      "يتقدم مجلس الإدارة بخالص الشكر لجميع الموظفين على جهودهم ويثق بمستقبل الشركة."]

FIXTURES = {
    "zh_body_en_headings.pdf": ("china-s", ["公司年度报告", "Financial Review", "Risk Factors", "Business Outlook"], ZH),
    "ja_report.pdf": ("japan", ["年次報告書", "財務ハイライト", "リスク要因", "今後の見通し"], JA),
    "hi_report.pdf": ("figo", ["वार्षिक रिपोर्ट", "वित्तीय समीक्षा", "जोखिम कारक", "भविष्य की दिशा"], HI),
    "ar_report.pdf": ("figo", ["التقرير السنوي", "المراجعة المالية", "عوامل المخاطر", "التوقعات المستقبلية"], AR),
}


def build(font, headings, body):
    doc = pymupdf.Document()
    buffer = pymupdf.Font(font).buffer
    for heading in headings:
        page = doc.new_page(width=595, height=842)
        page.insert_font(fontname="f", fontbuffer=buffer)
        page.insert_text((72, 90), heading, fontsize=20, fontname="f")
        for row, line in enumerate(body):
            page.insert_text((72, 130 + 16 * row), line, fontsize=11, fontname="f")
    doc.subset_fonts()
    return doc.tobytes(garbage=4, deflate=True, no_new_id=True)


if __name__ == "__main__":
    for name, (font, headings, body) in FIXTURES.items():
        (HERE / name).write_bytes(build(font, headings, body))
        print(name)
