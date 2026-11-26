from concurrent.futures import ThreadPoolExecutor
import glob
import html
import io
import os
import pickle
import re
import shutil
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.signal import savgol_filter
import streamlit as st
import streamlit.components.v1 as components
import xlrd

try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

st.set_page_config(page_title="ODCS Stiction Analysis Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        zoom: 0.90;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    /* 判定結果一覧用：横スクロール可能＆文字見切れ防止テーブル */
    .odcs-scroll-table-wrap {
        width: 100%;
        overflow-x: auto;
        overflow-y: auto;
        max-height: 440px;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        background: #ffffff;
    }
    .odcs-scroll-table {
        width: max-content;
        min-width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        white-space: nowrap;
    }
    .odcs-scroll-table th {
        position: sticky;
        top: 0;
        background-color: #f8fafc;
        color: #1e293b;
        font-weight: 600;
        text-align: left;
        padding: 7px 12px;
        border-bottom: 1px solid #cbd5e1;
        z-index: 1;
    }
    .odcs-scroll-table td {
        padding: 6px 12px;
        border-bottom: 1px solid #f1f5f9;
        color: #0f172a;
    }
    .odcs-scroll-table tr:hover td {
        background-color: #f8fafc;
    }

    /* ==========================================================
       個別波形グラフ一覧：β+Lot情報ヘッダー＆カードサイズ完全統一
       ========================================================== */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.odcs-lot-header-top) {
        background-color: #f8fafc !important;
        border-color: #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 8px 10px !important;
        height: 104px !important;
        min-height: 104px !important;
        max-height: 104px !important;
        overflow: hidden !important;
    }
    .odcs-lot-header-top {
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 4px;
    }
    .odcs-lot-header-title {
        font-size: 15.5px;
        font-weight: 700;
        color: #0f172a;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 2px;
    }
    .odcs-lot-header-sub {
        font-size: 11.5px;
        color: #475569;
        line-height: 1.3;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .odcs-ref-label-inline {
        font-size: 11.5px;
        font-weight: 600;
        color: #475569;
        padding-top: 6px;
        white-space: nowrap;
    }

    /* 個別グラフカード全体の高さ＆タイトル領域の高さを全カード完全統一 */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.odcs-card-title-box) {
        height: 318px !important;
        min-height: 318px !important;
        max-height: 318px !important;
        overflow: hidden;
    }
    .odcs-empty-card-box {
        height: 318px;
        border: 1px dashed #cbd5e1;
        border-radius: 8px;
        background: #f8fafc;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #94a3b8;
        font-size: 12.5px;
        box-sizing: border-box;
    }
    .odcs-card-title-box {
        height: 52px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        overflow: hidden;
        font-size: 12.5px;
        line-height: 1.35;
    }
    .odcs-card-line1 {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #0f172a;
    }
    .odcs-card-line2 {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .odcs-badge-ok {
        color: #16a34a;
        font-weight: 700;
    }
    .odcs-badge-warn {
        color: #d97706;
        font-weight: 700;
    }
    .odcs-badge-ng {
        color: #dc2626;
        font-weight: 700;
    }
    .odcs-val-pill {
        background: #f1f5f9;
        color: #1e293b;
        padding: 0px 5px;
        border-radius: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 11.5px;
        font-weight: 600;
    }
    .odcs-badge-ov {
        background: #dbeafe;
        color: #1d4ed8;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 10.5px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CACHE_FILE_NAME = ".odcs_cache_v12_models.pkl"
DEFAULT_GDRIVE_FOLDER_ID = "18z6yjQTeQWIdcCUCYID8mTqGgAbIa2Yr"
ROOT_SUBFOLDER_NAME = "Stickness Peak Test"
PRIORITY_MODEL_NAME = "M040-000013-P2BKKI"
CONTROL_LIMIT = 2.0
SPEC_LIMIT = 32.8

# ==========================================
# 多言語辞書（日本語 / 英語 / タイ語）
# ==========================================
LANG_OPTIONS = [
    "日本語 (Japanese)",
    "英語 (English)",
    "タイ語 (Thai)",
]

I18N = {
    "日本語 (Japanese)": {
        "app_title": "📊 ODCS 粘着自動判定ダッシュボード",
        "lang_label": "🌐 言語 (Language)",
        "btn_refresh": "🔄 データ更新（ドライブ再スキャン）",
        "model_select_label": "📦 品番選択 (Part No.)",
        "sidebar_filter_header": "🔍 絞り込みフィルター",
        "f_lot": "① β+Lot / 条件（初期直近3・最大5）",
        "f_trial": "② 測定回数（-1, -2, -3...）",
        "f_date": "③ 測定日 yymmdd",
        "f_hdd": "④ HDD No.",
        "trial_fmt": "-{x} 回目",
        "no_options": "選択肢なし",
        "all_selected": "すべて選択中 ({n}件)",
        "none_selected": "未選択 (0件)",
        "more_selected": "{first} ほか {rest}件選択中",
        "search_placeholder": "🔍 検索...",
        "sidebar_display_header": "🎨 一括表示・スケール設定",
        "only_ng": "NG / 工程管理外のみ絞り込み",
        "show_base": "全グラフに基準フィット線（点線）を表示",
        "auto_scale": "最大スケールに自動統一",
        "y_scale_cap": "Y軸上限 [mN・m]（スライダー / 数値入力）",
        "x_scale_cap": "X軸上限 [step]（スライダー / 数値入力）",
        "exp_data_settings": "⚙️ データ読込・ベースライン設定",
        "input_src": "データ入力元",
        "src_gdrive": "☁️ Googleドライブ自動読込（リンク共有）",
        "src_folder": "📁 ローカルフォルダ自動読込",
        "src_upload": "📤 ファイル / ZIP アップロード",
        "gdrive_folder_id": "Googleドライブ ルートフォルダID / URL",
        "gdrive_err": "Googleドライブ読込エラー（フォルダが「リンクを知っている全員」に共有されているか確認してください）: {err}",
        "folder_path": "対象フォルダパス",
        "folder_not_found": "フォルダが見つかりません",
        "uploader_label": "測定ファイル (.xls / .xlsx / .zip)",
        "base_mode_label": "基準ベースラインの決め方（全体初期値）",
        "base_auto": "自動（同Lot Pre-HT / 未処理はβ+量産 Pre-HT）",
        "base_custom": "指定LotのPre-HTを全データの基準に固定",
        "base_self": "単体データ外挿（基準カーブなし）",
        "custom_ref_sel": "基準にするLot (Pre-HT)",
        "pre_ht_eval_sel": "Pre-HTのピークも算出する条件",
        "info_no_data": "Googleドライブの「Stickness Peak Test」配下の各品番フォルダに測定ファイル（.xls / .xlsx / .zip）を入れるか、左サイドバーからファイルをアップロードしてください。",
        "metric_total": "判定総数",
        "metric_ok": "OK (≦ 2.0 mN・m)",
        "metric_warn": "工程管理外 (2.0〜32.8)",
        "metric_ng": "NG (> 32.8 / 異常)",
        "judge_disp_ok": "OK",
        "judge_disp_warn": "工程管理外",
        "judge_disp_ng": "NG",
        "unit_items": "{n} 件",
        "btn_csv": "📥 判定結果一覧をCSVダウンロード",
        "sec_summary": "📋 判定結果一覧",
        "warn_select_filter": "左サイドバーで各フィルター項目を1つ以上選択してください。",
        "warn_no_match": "選択されたフィルター条件に一致するデータがありません。",
        "sec_overlay": "📊 全選択データ 重ね描き比較グラフ",
        "ov_filter_title": "##### 🔍 比較グラフ専用フィルター",
        "ov_lot": "条件 (Lot)",
        "ov_hdd": "HDD No.",
        "ov_ref_mode": "基準ライン表示方式",
        "ov_ref_fit": "フィット済み基準線（点線）",
        "ov_ref_raw": "生データ Pre-HT（フィットなし）",
        "ov_ref_none": "基準線なし（評価線のみ）",
        "ov_tr": "測定回数",
        "ov_judge": "判定絞り込み",
        "judge_all": "すべて",
        "judge_ok": "OKのみ",
        "judge_warn": "管理外のみ",
        "judge_ng": "NGのみ",
        "judge_ng_warn": "NG・管理外",
        "tab_cards": "📈 個別波形グラフ一覧（HDD No. 揃え）",
        "tab_table": "📑 全データ詳細テーブル",
        "hdr_max": "最大",
        "hdr_ng": "NG",
        "hdr_warn": "管理外",
        "hdr_base": "基準:",
        "lot_ref_auto": "自動 ({auto_lbl})",
        "lot_ref_self": "単体外挿（基準なし）",
        "pop_ref_cap": "基準ライン個別変更（{hdd} の Pre-HT）",
        "ref_auto_opt": "列の基準設定に従う",
        "badge_in_ov": "📊 比較表示中",
        "no_data_card": "データなし",
        "axis_x": "回転角相当 [deg]",
        "axis_y": "トルク [mN・m]",
        "leg_title": "凡例一覧",
        "leg_clear": "選択解除",
        "leg_reset": "全表示",
        "leg_selected": "📌 選択中: ",
        "tbl_judge": "判定",
        "tbl_trial_col": "{t}回目 [mN・m]",
    },
    "英語 (English)": {
        "app_title": "📊 ODCS Automated Stiction Analysis Dashboard",
        "lang_label": "🌐 Language",
        "btn_refresh": "🔄 Refresh Data (Rescan Drive)",
        "model_select_label": "📦 Part No. (Model)",
        "sidebar_filter_header": "🔍 Filters",
        "f_lot": "① β+Lot / Condition (Default Latest 3, Max 5)",
        "f_trial": "② Trial No. (-1, -2, -3...)",
        "f_date": "③ Test Date (yymmdd)",
        "f_hdd": "④ HDD No.",
        "trial_fmt": "Trial -{x}",
        "no_options": "No options",
        "all_selected": "All selected ({n})",
        "none_selected": "None selected (0)",
        "more_selected": "{first} +{rest} more",
        "search_placeholder": "🔍 Search...",
        "sidebar_display_header": "🎨 Display & Scale Settings",
        "only_ng": "Show only NG / Out-of-Control rows",
        "show_base": "Show baseline fit curve (dotted) on all charts",
        "auto_scale": "Auto-unify to maximum scale",
        "y_scale_cap": "Y-Axis Max [mN·m] (Slider / Input)",
        "x_scale_cap": "X-Axis Max [step] (Slider / Input)",
        "exp_data_settings": "⚙️ Data Source & Baseline Settings",
        "input_src": "Data Source",
        "src_gdrive": "☁️ Auto-load from Google Drive (Shared Link)",
        "src_folder": "📁 Auto-load from Local Folder",
        "src_upload": "📤 Upload Files / ZIP",
        "gdrive_folder_id": "Google Drive Root Folder ID / URL",
        "gdrive_err": "Google Drive Error (Please verify the folder is shared with 'Anyone with the link'): {err}",
        "folder_path": "Target Folder Path",
        "folder_not_found": "Folder not found",
        "uploader_label": "Measurement Files (.xls / .xlsx / .zip)",
        "base_mode_label": "Global Default Baseline Mode",
        "base_auto": "Auto (Same Lot Pre-HT / Untreated uses β+ Mass Pre-HT)",
        "base_custom": "Fix Specified Lot Pre-HT as Global Reference",
        "base_self": "Self Extrapolation (No Reference Curve)",
        "custom_ref_sel": "Reference Lot (Pre-HT)",
        "pre_ht_eval_sel": "Also Evaluate Pre-HT Peaks for",
        "info_no_data": "Place measurement files (.xls / .xlsx / .zip) into each Part No. folder under 'Stickness Peak Test' on Google Drive, or upload files from the left sidebar.",
        "metric_total": "Total Evaluated",
        "metric_ok": "OK (≦ 2.0 mN·m)",
        "metric_warn": "Control Limit (2.0–32.8)",
        "metric_ng": "NG (> 32.8 / Error)",
        "judge_disp_ok": "OK",
        "judge_disp_warn": "Out of Control",
        "judge_disp_ng": "NG",
        "unit_items": "{n}",
        "btn_csv": "📥 Download Results as CSV",
        "sec_summary": "📋 Judgement Summary List",
        "warn_select_filter": "Please select at least one option in each filter on the left sidebar.",
        "warn_no_match": "No data matches the selected filter criteria.",
        "sec_overlay": "📊 Combined Overlay Comparison Chart",
        "ov_filter_title": "##### 🔍 Overlay Chart Filters",
        "ov_lot": "Condition (Lot)",
        "ov_hdd": "HDD No.",
        "ov_ref_mode": "Reference Line Mode",
        "ov_ref_fit": "Fitted Baseline (Dotted)",
        "ov_ref_raw": "Raw Pre-HT Data (Unfitted)",
        "ov_ref_none": "No Reference (Eval Only)",
        "ov_tr": "Trial No.",
        "ov_judge": "Judgement Filter",
        "judge_all": "All",
        "judge_ok": "OK",
        "judge_warn": "Control",
        "judge_ng": "NG",
        "judge_ng_warn": "NG+Ctrl",
        "tab_cards": "📈 Individual Waveform Cards (Aligned by HDD No.)",
        "tab_table": "📑 Full Detail Data Table",
        "hdr_max": "Max",
        "hdr_ng": "NG",
        "hdr_warn": "Ctrl",
        "hdr_base": "Ref:",
        "lot_ref_auto": "Auto ({auto_lbl})",
        "lot_ref_self": "Self Extrapolation",
        "pop_ref_cap": "Override Reference ({hdd} Pre-HT)",
        "ref_auto_opt": "Follow Lot Header Ref",
        "badge_in_ov": "📊 In Overlay",
        "no_data_card": "No Data",
        "axis_x": "Rotation Angle Equiv. [deg]",
        "axis_y": "Torque [mN·m]",
        "leg_title": "Legend",
        "leg_clear": "Deselect",
        "leg_reset": "Show All",
        "leg_selected": "📌 Selected: ",
        "tbl_judge": "Judgement",
        "tbl_trial_col": "Trial {t} [mN·m]",
    },
    "タイ語 (Thai)": {
        "app_title": "📊 แดชบอร์ดวิเคราะห์และตัดสินค่า Stiction อัตโนมัติ (ODCS)",
        "lang_label": "🌐 ภาษา (Language)",
        "btn_refresh": "🔄 อัปเดตข้อมูล (สแกนไดรฟ์ใหม่)",
        "model_select_label": "📦 เลือกรุ่นสินค้า (Part No.)",
        "sidebar_filter_header": "🔍 ตัวกรองข้อมูล (Filters)",
        "f_lot": "① β+Lot / เงื่อนไข (เริ่มต้น 3 ล่าสุด, สูงสุด 5)",
        "f_trial": "② ครั้งที่ทดสอบ (-1, -2, -3...)",
        "f_date": "③ วันที่ทดสอบ (yymmdd)",
        "f_hdd": "④ หมายเลข HDD (HDD No.)",
        "trial_fmt": "ครั้งที่ -{x}",
        "no_options": "ไม่มีตัวเลือก",
        "all_selected": "เลือกทั้งหมด ({n} รายการ)",
        "none_selected": "ไม่ได้เลือก (0 รายการ)",
        "more_selected": "{first} และอีก {rest} รายการ",
        "search_placeholder": "🔍 ค้นหา...",
        "sidebar_display_header": "🎨 ตั้งค่าการแสดงผลและสเกลกราฟ",
        "only_ng": "แสดงเฉพาะแถวที่เป็น NG / เกินเกณฑ์ควบคุม",
        "show_base": "แสดงเส้นอ้างอิงพื้นฐาน (เส้นจุด) ในทุกกราฟ",
        "auto_scale": "ปรับสเกลสูงสุดให้เท่ากันอัตโนมัติ",
        "y_scale_cap": "ค่าสูงสุดแกน Y [mN·m] (สไลเดอร์ / พิมพ์ตัวเลข)",
        "x_scale_cap": "ค่าสูงสุดแกน X [step] (สไลเดอร์ / พิมพ์ตัวเลข)",
        "exp_data_settings": "⚙️ ตั้งค่าการโหลดข้อมูลและเส้นอ้างอิง",
        "input_src": "แหล่งข้อมูล",
        "src_gdrive": "☁️ โหลดอัตโนมัติจาก Google Drive (แชร์ลิงก์)",
        "src_folder": "📁 โหลดจากโฟลเดอร์ในเครื่อง",
        "src_upload": "📤 อัปโหลดไฟล์ / ZIP",
        "gdrive_folder_id": "Google Drive Root Folder ID / URL",
        "gdrive_err": "เกิดข้อผิดพลาดในการโหลด Google Drive (โปรดตรวจสอบว่าตั้งค่าแชร์เป็น 'ทุกคนที่มีลิงก์' แล้ว): {err}",
        "folder_path": "ที่อยู่โฟลเดอร์เป้าหมาย",
        "folder_not_found": "ไม่พบโฟลเดอร์ที่ระบุ",
        "uploader_label": "ไฟล์ผลการวัด (.xls / .xlsx / .zip)",
        "base_mode_label": "วิธีเลือกเส้นอ้างอิงเริ่มต้น (Baseline)",
        "base_auto": "อัตโนมัติ (Pre-HT ของ Lot เดียวกัน / งานไม่ผ่านการอบใช้ β+ Mass)",
        "base_custom": "ใช้ Pre-HT ของ Lot ที่กำหนดเป็นเกณฑ์ทั้งหมด",
        "base_self": "คำนวณจากกราฟตัวเอง (ไม่มีเส้นอ้างอิง)",
        "custom_ref_sel": "เลือก Lot อ้างอิง (Pre-HT)",
        "pre_ht_eval_sel": "คำนวณค่าพีคของ Pre-HT ด้วยสำหรับเงื่อนไข",
        "info_no_data": "กรุณานำไฟล์ผลการวัด (.xls / .xlsx / .zip) ใส่ในโฟลเดอร์รุ่นสินค้าภายใต้ 'Stickness Peak Test' บน Google Drive หรืออัปโหลดทางแถบซ้าย",
        "metric_total": "จำนวนทั้งหมด",
        "metric_ok": "OK (≦ 2.0 mN·m)",
        "metric_warn": "เกินเกณฑ์ควบคุม (2.0–32.8)",
        "metric_ng": "NG (> 32.8 / ผิดปกติ)",
        "judge_disp_ok": "OK",
        "judge_disp_warn": "เกินเกณฑ์ควบคุม",
        "judge_disp_ng": "NG",
        "unit_items": "{n} รายการ",
        "btn_csv": "📥 ดาวน์โหลดตารางผลลัพธ์ (CSV)",
        "sec_summary": "📋 ตารางสรุปผลการตัดสิน",
        "warn_select_filter": "กรุณาเลือกตัวกรองอย่างน้อย 1 รายการในแถบด้านซ้าย",
        "warn_no_match": "ไม่พบข้อมูลที่ตรงกับเงื่อนไขตัวกรอง",
        "sec_overlay": "📊 กราฟเปรียบเทียบข้อมูลทั้งหมดแบบซ้อนทับ",
        "ov_filter_title": "##### 🔍 ตัวกรองเฉพาะกราฟเปรียบเทียบ",
        "ov_lot": "เงื่อนไข (Lot)",
        "ov_hdd": "หมายเลข HDD",
        "ov_ref_mode": "รูปแบบเส้นอ้างอิง",
        "ov_ref_fit": "เส้นอ้างอิงที่ทาบแล้ว (เส้นจุด)",
        "ov_ref_raw": "ข้อมูลดิบ Pre-HT (ไม่ทาบ)",
        "ov_ref_none": "ไม่แสดงเส้นอ้างอิง (เฉพาะเส้นหลัก)",
        "ov_tr": "ครั้งที่ทดสอบ",
        "ov_judge": "กรองตามผลตัดสิน",
        "judge_all": "ทั้งหมด",
        "judge_ok": "OK",
        "judge_warn": "เกินเกณฑ์",
        "judge_ng": "NG",
        "judge_ng_warn": "NG+เกินเกณฑ์",
        "tab_cards": "📈 รายการกราฟแยกแต่ละตัว (เรียงตาม HDD No.)",
        "tab_table": "📑 ตารางข้อมูลรายละเอียดทั้งหมด",
        "hdr_max": "สูงสุด",
        "hdr_ng": "NG",
        "hdr_warn": "เกินเกณฑ์",
        "hdr_base": "อ้างอิง:",
        "lot_ref_auto": "อัตโนมัติ ({auto_lbl})",
        "lot_ref_self": "คำนวณจากกราฟตัวเอง",
        "pop_ref_cap": "เปลี่ยนเส้นอ้างอิงเฉพาะตัว ({hdd} Pre-HT)",
        "ref_auto_opt": "ใช้ตามเกณฑ์ของคอลัมน์ด้านบน",
        "badge_in_ov": "📊 แสดงในกราฟเปรียบเทียบ",
        "no_data_card": "ไม่มีข้อมูล",
        "axis_x": "มุมการหมุนเทียบเท่า [deg]",
        "axis_y": "แรงบิด Torque [mN·m]",
        "leg_title": "คำอธิบายเส้น",
        "leg_clear": "ยกเลิกเลือก",
        "leg_reset": "แสดงทั้งหมด",
        "leg_selected": "📌 กำลังเลือก: ",
        "tbl_judge": "ผลตัดสิน",
        "tbl_trial_col": "ครั้งที่ {t} [mN·m]",
    },
}


def tr_txt(key: str, **kwargs) -> str:
    lang = st.session_state.get("ui_lang", LANG_OPTIONS[0])
    text = I18N.get(lang, I18N[LANG_OPTIONS[0]]).get(key, key)
    return text.format(**kwargs) if kwargs else text


def format_judge_label(judge_code: str) -> str:
    if judge_code == "OK":
        return tr_txt("judge_disp_ok")
    if judge_code == "OUT_OF_CONTROL":
        return tr_txt("judge_disp_warn")
    if judge_code == "NG_UL":
        return f"{tr_txt('judge_disp_ng')} (UL)"
    if judge_code == "NG":
        return tr_txt("judge_disp_ng")
    return judge_code


# ==========================================
# 1. ファイル名パーサー & 品番パス解決ヘルパー
# ==========================================
def resolve_model_from_rel_path(rel_path: str) -> str:
    """
    Stickness Peak Test 配下の相対パスから直下フォルダ名（品番）を抽出する。
    """
    norm = rel_path.replace("\\", "/").strip("/")
    parts = [p for p in norm.split("/") if p]
    if ROOT_SUBFOLDER_NAME in parts:
        idx = parts.index(ROOT_SUBFOLDER_NAME)
        if idx + 1 < len(parts) - 1:
            return parts[idx + 1]
    if len(parts) >= 2:
        return parts[0]
    return "未分類"


def parse_filename(filename: str, model_name: str = "未分類"):
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]

    hdd_match = re.search(r"([A-Za-z0-9]{4}\s+[A-Za-z0-9]{4})", name_no_ext)
    hdd_no = hdd_match.group(1) if hdd_match else "UNKNOWN_HDD"

    trial = 1
    if hdd_match:
        tr_match = re.match(r"^-(\d+)", name_no_ext[hdd_match.end():])
        if tr_match:
            trial = int(tr_match.group(1))

    if re.search(r"(Pre-HT|Pre_HT|熱処理前)", name_no_ext, re.IGNORECASE):
        heat_stage = "Pre-HT"
    elif re.search(r"((?<![A-Za-z])HT(?![A-Za-z])|Post-HT|Post_HT|熱処理後|熱処理跡)", name_no_ext, re.IGNORECASE):
        heat_stage = "HT"
    else:
        heat_stage = "不明"

    date_match = re.search(r"_(\d{6})$", name_no_ext)
    test_date = date_match.group(1) if date_match else "-"

    kb_match = re.search(r"(KB-\d{6}-\d+)", name_no_ext, re.IGNORECASE)
    is_untreated = False
    if kb_match:
        base_lot = kb_match.group(1).upper()
    elif "Blank" in name_no_ext:
        base_lot = "Blank"
    elif re.search(r"(未処理|Without|Whisout)", name_no_ext, re.IGNORECASE):
        base_lot = "未処理"
        is_untreated = True
    elif "TKR" in name_no_ext or "量産" in name_no_ext:
        base_lot = "β+量産品(TKR)"
    else:
        cleaned = name_no_ext.replace(hdd_no, "") if hdd_match else name_no_ext
        cleaned = re.sub(r"-\d+|(Pre-HT|Post-HT|HT|熱処理前|熱処理後|熱処理跡|\(初回UL異常\))|_\d{6}$", "", cleaned, flags=re.IGNORECASE).strip("_- ")
        base_lot = cleaned if cleaned else "その他Lot"

    lot_name = f"未処理 ({heat_stage})" if (is_untreated and heat_stage in ("Pre-HT", "HT")) else base_lot
    ul_error = bool(re.search(r"(初回UL異常|UL_ERR|UL異常)", name_no_ext, re.IGNORECASE))
    priority = 0 if "熱処理跡" in name_no_ext else 1

    return {
        "model_name": model_name,
        "file_name": base,
        "lot_name": lot_name,
        "base_lot": base_lot,
        "is_untreated": is_untreated,
        "HDD_No": hdd_no,
        "trial": trial,
        "heat_stage": heat_stage,
        "test_date": test_date,
        "ul_error": ul_error,
        "priority": priority,
    }


# ==========================================
# 2. 「生」シートからの直接高速計算 & Googleドライブ（リンク共有）読込エンジン
# ==========================================
def load_torque_from_xls(source):
    xls = pd.ExcelFile(source if isinstance(source, str) else io.BytesIO(source))
    raw_sheet = "生" if "生" in xls.sheet_names else next((s for s in xls.sheet_names if "生" in s), None)

    if raw_sheet is not None:
        df_raw = pd.read_excel(xls, sheet_name=raw_sheet, usecols="A:B")
        raw_b = pd.to_numeric(df_raw.iloc[:, 1], errors="coerce").fillna(0.0).values
        trq_all = raw_b / 100000.0 * 0.95
        s_cand = np.where(trq_all > 0.20)[0]
        e_cand = np.where(trq_all > 0.10)[0]
        if len(s_cand) == 0 or len(e_cand) == 0 or e_cand[-1] <= s_cand[0] + 10:
            return np.array([], dtype=np.float32)
        trq_arr = trq_all[int(s_cand[0]) : int(e_cand[-1]) + 1]
    else:
        sheet = "補正１ (0.2)" if "補正１ (0.2)" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet, usecols="A:G")
        g_col = pd.to_numeric(df.iloc[:, 6], errors="coerce").dropna()
        nz = g_col[g_col > 0.1]
        if len(nz) == 0:
            return np.array([], dtype=np.float32)
        trq_arr = pd.to_numeric(df.iloc[: int(nz.index[-1]) + 1, 6], errors="coerce").fillna(0.0).values

    if len(trq_arr) > 80:
        trq_arr = trq_arr[:-25]

    return trq_arr.astype(np.float32)


def _read_single_file_task(args):
    file_path, model_name = args
    meta = parse_filename(file_path, model_name=model_name)
    meta["torque"] = load_torque_from_xls(file_path)
    meta["mtime"] = os.path.getmtime(file_path)
    return file_path, meta


def extract_gdrive_folder_id(raw_input: str) -> str:
    s = (raw_input or "").strip()
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)
    m2 = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", s)
    if m2:
        return m2.group(1)
    return s


def register_item_by_model(models_data: dict, meta: dict):
    model = meta.get("model_name", "未分類")
    if model not in models_data:
        models_data[model] = {}
    parsed_data = models_data[model]
    key = (meta["base_lot"], meta["HDD_No"], meta["heat_stage"], int(meta["trial"]))
    if key not in parsed_data or parsed_data[key]["priority"] < meta["priority"]:
        parsed_data[key] = meta


@st.cache_data(show_spinner="Downloading data from Google Drive...")
def load_raw_dataset_from_gdrive_public(folder_id_or_url: str, refresh_token: int):
    """
    Stiction Peak Analyzer (root) -> Stickness Peak Test -> [品番フォルダ一覧]
    配下の全ファイル・サブフォルダ・ZIPを品番ごとに自動分類して読み込む
    """
    if not HAS_GDOWN:
        return {}, [], "gdown library is not installed."

    folder_id = extract_gdrive_folder_id(folder_id_or_url)
    if not folder_id:
        return {}, [], "Folder ID is empty."

    base_tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gdrive_sync_tmp", folder_id)
    if os.path.exists(base_tmp_dir):
        shutil.rmtree(base_tmp_dir, ignore_errors=True)
    os.makedirs(base_tmp_dir, exist_ok=True)

    try:
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(
            url=folder_url,
            output=base_tmp_dir,
            quiet=True,
            use_cookies=False,
        )

        # Stickness Peak Test フォルダを探す（無ければ base_tmp_dir 直下を対象にする）
        scan_root = base_tmp_dir
        for root, dirs, _ in os.walk(base_tmp_dir):
            if ROOT_SUBFOLDER_NAME in dirs:
                scan_root = os.path.join(root, ROOT_SUBFOLDER_NAME)
                break

        discovered_models = sorted([
            d for d in os.listdir(scan_root)
            if os.path.isdir(os.path.join(scan_root, d)) and not d.startswith(".")
        ])

        models_data = {m: {} for m in discovered_models}

        for root, _, files in os.walk(scan_root):
            rel_dir = os.path.relpath(root, scan_root)
            if rel_dir == ".":
                model_name = "未分類"
            else:
                model_name = rel_dir.replace("\\", "/").split("/")[0]

            for fname in sorted(files):
                if fname.startswith("~$"):
                    continue
                fpath = os.path.join(root, fname)
                mtime_ts = os.path.getmtime(fpath)

                if fname.lower().endswith(".zip"):
                    with zipfile.ZipFile(fpath, "r") as z:
                        for member in z.namelist():
                            if member.lower().endswith((".xls", ".xlsx")) and not os.path.basename(member).startswith("~$"):
                                eff_model = model_name
                                if eff_model == "未分類" and "/" in member.strip("/"):
                                    eff_model = member.strip("/").split("/")[0]
                                meta = parse_filename(member, model_name=eff_model)
                                meta["torque"] = load_torque_from_xls(z.read(member))
                                meta["mtime"] = mtime_ts
                                register_item_by_model(models_data, meta)
                elif fname.lower().endswith((".xls", ".xlsx")):
                    meta = parse_filename(fname, model_name=model_name)
                    meta["torque"] = load_torque_from_xls(fpath)
                    meta["mtime"] = mtime_ts
                    register_item_by_model(models_data, meta)

        all_models = sorted(models_data.keys())
        return models_data, all_models, None
    except Exception as e:
        return {}, [], str(e)


# ==========================================
# 3. 2区間フィッティング（前半急勾配＋後半定常域）粘着判定アルゴリズム
#    判定基準:
#      - NG: > 32.8 mN・m (または初回UL異常)
#      - 工程管理外 (OUT_OF_CONTROL): > 2.0 mN・m かつ <= 32.8 mN・m
#      - OK: <= 2.0 mN・m
# ==========================================
def analyze_stiction(
    torque: np.ndarray,
    ul_error: bool,
    ref_torque: np.ndarray = None,
    noise_thresh: float = 0.30,
    drop_thresh: float = 0.50,
    control_limit: float = CONTROL_LIMIT,
    spec_limit: float = SPEC_LIMIT,
):
    n = len(torque)
    if n < 50:
        return {"stiction_peak": 0.0, "peak_idx": None, "judge": "データ不足"}

    win = min(15, n if n % 2 == 1 else n - 1)
    torque_smooth = savgol_filter(torque, window_length=win, polyorder=2) if win >= 5 else torque.copy()

    start_s = 300
    end_s = min(1500, n - 150)
    if end_s <= start_s + 20:
        return {"stiction_peak": 0.0, "peak_idx": None, "judge": "NG_UL" if ul_error else "OK"}

    s_indices = np.arange(start_s, end_s + 1)
    rough_peak_idx = start_s + int(np.argmax(torque_smooth[start_s : end_s + 1]))

    post_end = min(rough_peak_idx + 350, 1750, n - 1)
    bot_idx = (rough_peak_idx + 1 + int(np.argmin(torque_smooth[rough_peak_idx + 1 : post_end + 1]))) if post_end > rough_peak_idx else rough_peak_idx

    st_start = max(bot_idx + 30, 1350)
    st_end = min(1950, n - 1)
    if st_end - st_start < 15:
        st_start = bot_idx + 30
        st_end = min(1950, n - 1)

    idx_all = np.arange(n, dtype=np.float32)

    if ref_torque is not None and len(ref_torque) > 50 and (st_end - st_start) >= 15:
        r_n = len(ref_torque)
        r_win = min(15, r_n if r_n % 2 == 1 else r_n - 1)
        ref_smooth = savgol_filter(ref_torque, window_length=r_win, polyorder=2) if r_win >= 5 else ref_torque.copy()
        ref_idx = np.arange(r_n, dtype=np.float32)

        idx_tail = np.arange(st_start, st_end + 1, dtype=np.float32)
        trq_tail = torque_smooth[st_start : st_end + 1]

        slope_start = 40
        slope_end = min(320, rough_peak_idx - 20, n - 1)
        has_front_slope = (slope_end - slope_start) >= 30

        if has_front_slope:
            idx_front = np.arange(slope_start, slope_end + 1, dtype=np.float32)
            trq_front = torque_smooth[slope_start : slope_end + 1]
            idx_fit = np.concatenate([idx_front, idx_tail])
            trq_fit = np.concatenate([trq_front, trq_tail])
            weights = np.concatenate([
                np.full(len(idx_front), 1.2, dtype=np.float32),
                np.full(len(idx_tail), 1.0, dtype=np.float32),
            ])
            w_sum = float(np.sum(weights))
        else:
            idx_fit = idx_tail
            trq_fit = trq_tail
            weights = np.ones(len(idx_tail), dtype=np.float32)
            w_sum = float(len(idx_tail))

        best_rmse, x_shift_opt, y_shift_opt = 1e9, 0.0, 0.0
        for dx in np.linspace(-250, 250, 101):
            shifted_ref_idx = ref_idx + dx
            if shifted_ref_idx[0] <= idx_fit[0] and shifted_ref_idx[-1] >= idx_fit[-1]:
                interp_ref = np.interp(idx_fit, shifted_ref_idx, ref_smooth)
                diff = trq_fit - interp_ref
                dy = float(np.sum(weights * diff) / w_sum)
                rmse = float(np.sqrt(np.sum(weights * ((diff - dy) ** 2)) / w_sum))
                if rmse < best_rmse:
                    best_rmse, x_shift_opt, y_shift_opt = rmse, dx, dy

        baseline_curve = np.interp(idx_all, ref_idx + x_shift_opt, ref_smooth) + y_shift_opt
    else:
        if (st_end - st_start) >= 20:
            idx_st = np.arange(st_start, st_end + 1, dtype=np.float32)
            baseline_curve = np.polyval(np.polyfit(idx_st, torque_smooth[st_start : st_end + 1], deg=2), idx_all)
        else:
            baseline_curve = np.full_like(torque_smooth, torque_smooth[bot_idx])

    delta_torque = torque_smooth - baseline_curve
    valid_peak_val, valid_peak_idx = 0.0, None

    for idx in s_indices[np.argsort(delta_torque[s_indices])[::-1]]:
        dt_val = float(delta_torque[idx])
        if dt_val <= noise_thresh:
            break
        cur_trq = float(torque_smooth[idx])
        sharp_end = min(idx + 140, n - 1)
        valley_end = min(idx + 250, n - 1)

        if sharp_end > idx and valley_end > idx:
            if (cur_trq - float(np.min(torque_smooth[idx + 1 : sharp_end + 1])) >= drop_thresh) and (
                dt_val - float(np.min(delta_torque[idx + 1 : valley_end + 1])) >= drop_thresh
            ):
                valley_idx = idx + 1 + int(np.argmin(torque_smooth[idx + 1 : valley_end + 1]))
                pv_start, pv_end = valley_idx + 20, min(valley_idx + 200, n - 1)
                if pv_end - pv_start >= 5 and float(np.max(torque_smooth[pv_start : pv_end + 1])) >= float(torque_smooth[valley_idx]) - 0.10:
                    valid_peak_val, valid_peak_idx = dt_val, int(idx)
                    break

    if ul_error:
        judge = "NG_UL"
    elif valid_peak_val > spec_limit:
        judge = "NG"
    elif valid_peak_val > control_limit:
        judge = "OUT_OF_CONTROL"
    else:
        judge = "OK"

    return {
        "stiction_peak": round(valid_peak_val, 3),
        "peak_idx": valid_peak_idx,
        "judge": judge,
        "baseline_curve": baseline_curve.astype(np.float32),
    }


# ==========================================
# 4. ローカル / アップロード読込＆一括解析エンジン
# ==========================================
@st.cache_data(show_spinner="Loading local folder data...")
def load_raw_dataset_from_folder(folder_path: str, refresh_token: int):
    scan_root = os.path.join(folder_path, ROOT_SUBFOLDER_NAME) if os.path.isdir(os.path.join(folder_path, ROOT_SUBFOLDER_NAME)) else folder_path

    discovered_models = sorted([
        d for d in os.listdir(scan_root)
        if os.path.isdir(os.path.join(scan_root, d)) and not d.startswith(".")
    ])

    file_paths = sorted([
        p for pat in ("*.xls", "*.xlsx")
        for p in glob.glob(os.path.join(scan_root, "**", pat), recursive=True)
        if not os.path.basename(p).startswith("~$")
    ])

    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CACHE_FILE_NAME)
    file_cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                file_cache = pickle.load(f)
        except Exception:
            file_cache = {}

    current_set = set(file_paths)
    to_read = []
    for p in file_paths:
        rel_dir = os.path.relpath(os.path.dirname(p), scan_root)
        m_name = "未分類" if rel_dir == "." else rel_dir.replace("\\", "/").split("/")[0]
        if p not in file_cache or file_cache[p].get("mtime") != os.path.getmtime(p) or file_cache[p].get("model_name") != m_name:
            to_read.append((p, m_name))

    for dk in [k for k in file_cache if k not in current_set]:
        del file_cache[dk]

    if to_read:
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            for p, meta in executor.map(_read_single_file_task, to_read):
                file_cache[p] = meta
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(file_cache, f)
        except Exception:
            pass

    models_data = {m: {} for m in discovered_models}
    for p in file_paths:
        if p in file_cache:
            register_item_by_model(models_data, file_cache[p].copy())

    return models_data, sorted(models_data.keys())


@st.cache_data(show_spinner="Reading uploaded files...")
def load_raw_dataset_from_uploads(file_tuples):
    models_data = {}
    for fname, fbytes in file_tuples:
        if fname.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(fbytes), "r") as z:
                for member in z.namelist():
                    if member.lower().endswith((".xls", ".xlsx")) and not os.path.basename(member).startswith("~$"):
                        m_name = resolve_model_from_rel_path(member)
                        meta = parse_filename(member, model_name=m_name)
                        meta["torque"] = load_torque_from_xls(z.read(member))
                        register_item_by_model(models_data, meta)
        else:
            m_name = resolve_model_from_rel_path(fname)
            meta = parse_filename(fname, model_name=m_name)
            meta["torque"] = load_torque_from_xls(fbytes)
            register_item_by_model(models_data, meta)
    return models_data, sorted(models_data.keys())


def select_default_model(models_data: dict, available_models: list) -> str:
    """
    測定日(yymmdd)が最新の品番を初期選択。同率の場合は M040-000013-P2BKKI を優先する。
    """
    if not available_models:
        return ""
    if len(available_models) == 1:
        return available_models[0]

    best_model = available_models[0]
    best_score = ("000000", 0)

    for m in available_models:
        pdata = models_data.get(m, {})
        valid_dates = [
            str(item.get("test_date", ""))
            for item in pdata.values()
            if str(item.get("test_date", "")).isdigit() and len(str(item.get("test_date", ""))) == 6
        ]
        max_date = max(valid_dates) if valid_dates else "000000"
        prio_flag = 1 if m == PRIORITY_MODEL_NAME else 0
        score = (max_date, prio_flag)
        if score > best_score:
            best_score = score
            best_model = m

    return best_model


def run_batch_analysis(
    raw_parsed_data: dict,
    baseline_mode: str,
    custom_ref_lot: str,
    pre_ht_eval_lots: list,
    lot_ref_overrides: dict = None,
):
    if lot_ref_overrides is None:
        lot_ref_overrides = {}

    parsed_data = {k: v.copy() for k, v in raw_parsed_data.items()}
    all_base_lots = sorted({k[0] for k in parsed_data.keys()})
    beta_lots = [l for l in all_base_lots if l == "β+量産品(TKR)" or l.startswith("KB-")]
    lots_with_ht = {k[0] for k in parsed_data.keys() if k[2] == "HT"}

    results = []
    for key, item in sorted(parsed_data.items()):
        base_lot, hdd, stage, trial = key
        disp_lot = item["lot_name"]

        is_target = (
            (stage == "HT")
            or (stage == "Pre-HT" and (disp_lot in pre_ht_eval_lots or base_lot in pre_ht_eval_lots))
            or (stage == "Pre-HT" and base_lot not in lots_with_ht)
        )
        if not is_target:
            continue

        lot_ov = lot_ref_overrides.get(disp_lot)
        if lot_ov == "__SELF__":
            cand_lots = []
            allow_same_lot_pre = False
        elif lot_ov and lot_ov != "__AUTO__":
            cand_lots = [lot_ov]
            allow_same_lot_pre = True
        elif baseline_mode == "self":
            cand_lots = []
            allow_same_lot_pre = False
        elif baseline_mode == "custom" and custom_ref_lot:
            cand_lots = [custom_ref_lot]
            allow_same_lot_pre = True
        else:
            cand_lots = (beta_lots + ["Blank"]) if base_lot == "未処理" else ([base_lot] + [b for b in beta_lots if b != base_lot] + ["Blank"])
            allow_same_lot_pre = False

        ref_trq, ref_source_label = None, "単体外挿"
        for c_lot in cand_lots:
            if stage == "Pre-HT" and c_lot == base_lot and not allow_same_lot_pre:
                continue
            for tr_cand in (trial, 1):
                if (c_lot, hdd, "Pre-HT", tr_cand) in parsed_data:
                    ref_trq = parsed_data[(c_lot, hdd, "Pre-HT", tr_cand)]["torque"]
                    ref_source_label = f"{c_lot} Pre-HT"
                    break
            if ref_trq is not None:
                break

        res = analyze_stiction(item["torque"], item["ul_error"], ref_trq)
        item["analysis"] = res
        results.append({
            "品番": item.get("model_name", "-"),
            "β+Lot / 条件": disp_lot,
            "base_lot": base_lot,
            "HDD No.": hdd,
            "熱処理": stage,
            "回数": int(trial),
            "粘着ピーク [mN・m]": res["stiction_peak"],
            "judge_code": res["judge"],
            "判定": format_judge_label(res["judge"]),
            "基準ベースライン": ref_source_label,
            "測定日": item["test_date"],
            "mtime": float(item.get("mtime", 0.0)),
            "ファイル名": item["file_name"],
        })

    return parsed_data, pd.DataFrame(results)


def get_recent_lots(df_res: pd.DataFrame, n: int = 3) -> list:
    if df_res.empty:
        return []
    lot_scores = []
    for lot, grp in df_res.groupby("β+Lot / 条件"):
        valid_dates = [str(d) for d in grp["測定日"].unique() if str(d).isdigit() and len(str(d)) == 6]
        max_date = max(valid_dates) if valid_dates else "000000"
        max_mtime = float(grp["mtime"].max()) if "mtime" in grp.columns else 0.0
        lot_scores.append((max_date, max_mtime, str(lot)))

    lot_scores.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [x[2] for x in lot_scores[:n]]


# ==========================================
# 5. グラフ生成関数
# ==========================================
def build_combined_overlay_chart(
    parsed_data: dict,
    df_overlay: pd.DataFrame,
    ov_lots: list,
    ov_trials: list,
    ov_ref_mode: str = "fit",
    x_upper: int = 2450,
    y_upper: float = 56.0,
):
    fig = go.Figure()
    palette = (
        px.colors.qualitative.Dark24
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Alphabet
    )
    color_idx = 0
    displayed_cards = set()

    for lot in ov_lots:
        lot_df = df_overlay[df_overlay["β+Lot / 条件"] == lot]
        hdds = sorted(lot_df["HDD No."].unique())

        if lot == "未処理 (Pre-HT)":
            eval_stages = ["Pre-HT"]
        elif lot == "未処理 (HT)":
            eval_stages = ["HT"]
        else:
            eval_stages = ["HT"] if any(lot_df["熱処理"] == "HT") else ["Pre-HT"]

        for hdd in hdds:
            sub_hdd = lot_df[lot_df["HDD No."] == hdd]
            base_lot = sub_hdd["base_lot"].iloc[0]
            pair_color = palette[color_idx % len(palette)]
            color_idx += 1
            has_any_trace = False

            for tr in sorted(int(t) for t in ov_trials):
                tr_suffix = f"-{tr}" if (len(ov_trials) > 1 or tr != 1) else ""

                for stg in eval_stages:
                    if (base_lot, hdd, stg, tr) not in parsed_data:
                        continue
                    d = parsed_data[(base_lot, hdd, stg, tr)]
                    an = d.get("analysis")
                    n_pts = len(d["torque"])
                    if n_pts == 0:
                        continue

                    has_any_trace = True
                    step = max(1, n_pts // 500)
                    idx_x = np.arange(0, n_pts, step)

                    pk_txt = f" ({an['stiction_peak']:.2f})" if an else ""
                    lot_prefix = f"[{lot}] " if len(ov_lots) > 1 else ""
                    trace_name = f"{lot_prefix}{hdd} {stg}{tr_suffix}{pk_txt}"

                    fig.add_trace(go.Scatter(
                        x=idx_x,
                        y=d["torque"][idx_x],
                        mode="lines",
                        name=trace_name,
                        legendgroup=trace_name,
                        line=dict(color=pair_color, width=2.2, dash="solid"),
                        opacity=0.95,
                        showlegend=False,
                    ))

                    if ov_ref_mode == "fit" and an and "baseline_curve" in an:
                        ref_trace_name = f"{lot_prefix}{hdd} 基準(Fit){tr_suffix}"
                        fig.add_trace(go.Scatter(
                            x=idx_x,
                            y=an["baseline_curve"][idx_x],
                            mode="lines",
                            name=ref_trace_name,
                            legendgroup=ref_trace_name,
                            line=dict(color=pair_color, width=1.6, dash="dot"),
                            opacity=0.82,
                            showlegend=False,
                        ))
                    elif ov_ref_mode == "raw" and stg == "HT" and (base_lot, hdd, "Pre-HT", tr) in parsed_data:
                        d_pre = parsed_data[(base_lot, hdd, "Pre-HT", tr)]
                        n_pre = len(d_pre["torque"])
                        if n_pre > 0:
                            step_pre = max(1, n_pre // 500)
                            idx_pre = np.arange(0, n_pre, step_pre)
                            raw_trace_name = f"{lot_prefix}{hdd} Pre-HT(Raw){tr_suffix}"
                            fig.add_trace(go.Scatter(
                                x=idx_pre,
                                y=d_pre["torque"][idx_pre],
                                mode="lines",
                                name=raw_trace_name,
                                legendgroup=raw_trace_name,
                                line=dict(color=pair_color, width=1.6, dash="dot"),
                                opacity=0.82,
                                showlegend=False,
                            ))

                    if an and an.get("peak_idx") is not None and an.get("stiction_peak", 0.0) > 0:
                        p_idx = an["peak_idx"]
                        fig.add_trace(go.Scatter(
                            x=[p_idx],
                            y=[d["torque"][p_idx]],
                            mode="markers",
                            name=trace_name,
                            legendgroup=trace_name,
                            marker=dict(color=pair_color, size=7, symbol="diamond"),
                            hovertemplate=f"{trace_name}<br>Peak: +{an['stiction_peak']:.2f} mNm<extra></extra>",
                            showlegend=False,
                        ))

            if has_any_trace:
                displayed_cards.add((lot, hdd))

    max_tick = int(np.ceil(x_upper / 500.0)) * 500
    tick_vals = list(range(0, max_tick + 1, 500))
    tick_texts = [f"{v / 1000.0:g}" for v in tick_vals]

    fig.update_layout(
        height=395,
        margin=dict(l=42, r=12, t=12, b=36),
        clickmode="event",
        hovermode="closest",
        xaxis=dict(
            title=tr_txt("axis_x"),
            title_font=dict(size=11),
            range=[-50, x_upper],
            tickvals=tick_vals,
            ticktext=tick_texts,
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="#eeeeee",
        ),
        yaxis=dict(
            title=tr_txt("axis_y"),
            title_font=dict(size=11),
            range=[0, y_upper],
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="#eeeeee",
        ),
        showlegend=False,
        plot_bgcolor="#FFFFF5",
        paper_bgcolor="white",
    )
    return fig, displayed_cards


def render_interactive_overlay_chart(fig: go.Figure):
    fig_json = pio.to_json(fig)
    leg_title_js = html.escape(tr_txt("leg_title"))
    leg_clear_js = html.escape(tr_txt("leg_clear"))
    leg_reset_js = html.escape(tr_txt("leg_reset"))
    leg_sel_js = html.escape(tr_txt("leg_selected"))

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: transparent;
        overflow: hidden;
      }}
      #container {{
        display: flex;
        width: 100%;
        height: 400px;
        gap: 10px;
      }}
      #chart-area {{
        flex: 1 1 auto;
        min-width: 0;
        position: relative;
        height: 396px;
      }}
      #plotly-div {{
        width: 100%;
        height: 396px;
      }}
      #plotly-div.pointer-cursor .nsewdrag {{
        cursor: pointer !important;
      }}
      #legend-panel {{
        flex: 0 0 265px;
        height: 396px;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        background: #ffffff;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }}
      #legend-header {{
        padding: 6px 8px;
        background: #f8fafc;
        border-bottom: 1px solid #e2e8f0;
        font-size: 11.5px;
        font-weight: 700;
        color: #334155;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 4px;
      }}
      .hdr-btns {{
        display: flex;
        gap: 4px;
        flex-shrink: 0;
      }}
      .panel-btn {{
        font-size: 10.5px;
        padding: 2px 6px;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        background: #fff;
        color: #475569;
        cursor: pointer;
        white-space: nowrap;
      }}
      .panel-btn:hover {{
        background: #f1f5f9;
      }}
      #clear-btn {{
        border-color: #f59e0b;
        color: #92400e;
        background: #fffbeb;
      }}
      #clear-btn:hover {{
        background: #fef3c7;
      }}
      #selected-detail {{
        display: none;
        padding: 6px 10px;
        background: #fef9c3;
        border-bottom: 1px solid #fde047;
        font-size: 11.5px;
        font-weight: 700;
        color: #713f12;
        line-height: 1.35;
        cursor: pointer;
      }}
      #selected-detail:hover {{
        background: #fef08a;
      }}
      #legend-list {{
        flex: 1 1 auto;
        overflow-y: auto;
        padding: 4px;
      }}
      .legend-item {{
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 5px 7px;
        border-radius: 4px;
        font-size: 11.5px;
        color: #1e293b;
        cursor: pointer;
        border: 1px solid transparent;
        user-select: none;
        transition: background 0.12s;
      }}
      .legend-item:hover {{
        background: #f1f5f9;
      }}
      .legend-item.highlighted {{
        background: #fef08a;
        border-color: #eab308;
        font-weight: 700;
        color: #713f12;
        box-shadow: 0 1px 3px rgba(234, 179, 8, 0.25);
      }}
      .legend-item.hidden-trace {{
        opacity: 0.4;
        text-decoration: line-through;
      }}
      .swatch {{
        width: 14px;
        height: 3px;
        flex-shrink: 0;
        border-radius: 2px;
      }}
      .swatch.dashed {{
        border-top: 2.5px dashed;
        height: 0;
        background: transparent !important;
      }}
      .swatch.dotted {{
        border-top: 2.5px dotted;
        height: 0;
        background: transparent !important;
      }}
      .item-label {{
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
    </style>
    </head>
    <body>
    <div id="container">
      <div id="chart-area">
        <div id="plotly-div"></div>
      </div>
      <div id="legend-panel">
        <div id="legend-header">
          <span>{leg_title_js}</span>
          <div class="hdr-btns">
            <button id="clear-btn" class="panel-btn" title="選択ハイライトのみ解除">{leg_clear_js}</button>
            <button id="reset-btn" class="panel-btn" title="ハイライト・非表示をリセット">{leg_reset_js}</button>
          </div>
        </div>
        <div id="selected-detail" title="クリックで選択解除"></div>
        <div id="legend-list"></div>
      </div>
    </div>

    <script>
      const figData = {fig_json};
      const gd = document.getElementById('plotly-div');
      const legendList = document.getElementById('legend-list');
      const selectedDetail = document.getElementById('selected-detail');
      const clearBtn = document.getElementById('clear-btn');
      const resetBtn = document.getElementById('reset-btn');
      const selPrefix = "{leg_sel_js}";

      const uniqueSeries = [];
      const seenNames = new Set();
      figData.data.forEach((tr) => {{
        if (tr.mode === 'lines' && tr.name && !seenNames.has(tr.name)) {{
          seenNames.add(tr.name);
          uniqueSeries.push({{
            name: tr.name,
            color: (tr.line && tr.line.color) ? tr.line.color : '#333',
            dash: (tr.line && tr.line.dash) ? tr.line.dash : 'solid'
          }});
        }}
      }});

      const itemEls = new Map();
      const hiddenSet = new Set();
      let activeHighlightName = null;
      let justClickedTrace = false;

      uniqueSeries.forEach((s) => {{
        const div = document.createElement('div');
        div.className = 'legend-item';
        div.title = s.name;

        const sw = document.createElement('span');
        sw.className = 'swatch' + (s.dash === 'dash' ? ' dashed' : (s.dash === 'dot' ? ' dotted' : ''));
        if (s.dash === 'dash' || s.dash === 'dot') {{
          sw.style.borderColor = s.color;
        }} else {{
          sw.style.backgroundColor = s.color;
        }}

        const lbl = document.createElement('span');
        lbl.className = 'item-label';
        lbl.textContent = s.name;

        div.appendChild(sw);
        div.appendChild(lbl);
        legendList.appendChild(div);
        itemEls.set(s.name, div);

        div.addEventListener('click', () => {{
          if (hiddenSet.has(s.name)) {{
            hiddenSet.delete(s.name);
            div.classList.remove('hidden-trace');
          }}
          if (activeHighlightName === s.name) {{
            clearHighlightOnly();
          }} else {{
            selectAndHighlight(s.name, false);
          }}
        }});

        div.addEventListener('dblclick', (e) => {{
          e.stopPropagation();
          if (hiddenSet.has(s.name)) {{
            hiddenSet.delete(s.name);
            div.classList.remove('hidden-trace');
          }} else {{
            hiddenSet.add(s.name);
            div.classList.add('hidden-trace');
            if (activeHighlightName === s.name) {{
              activeHighlightName = null;
              selectedDetail.style.display = 'none';
            }}
          }}
          applyPlotStyles();
        }});
      }});

      const origWidths = figData.data.map(t => (t.line && t.line.width) ? t.line.width : 1.8);
      const origOpacities = figData.data.map(t => (t.opacity !== undefined) ? t.opacity : 0.9);

      function applyPlotStyles() {{
        const vis = figData.data.map(t => hiddenSet.has(t.name) ? false : true);
        const widths = figData.data.map((t, i) => {{
          if (activeHighlightName && t.name === activeHighlightName) {{
            return Math.max(origWidths[i] + 2.4, 4.2);
          }}
          return origWidths[i];
        }});
        const opacities = figData.data.map((t, i) => {{
          if (activeHighlightName) {{
            return (t.name === activeHighlightName) ? 1.0 : 0.20;
          }}
          return origOpacities[i];
        }});

        Plotly.restyle(gd, {{
          'visible': vis,
          'line.width': widths,
          'opacity': opacities
        }});
      }}

      function selectAndHighlight(targetName, scrollIntoViewFlag = true) {{
        activeHighlightName = targetName;
        selectedDetail.textContent = selPrefix + targetName + ' ✕';
        selectedDetail.style.display = 'block';

        itemEls.forEach((el, name) => {{
          if (name === targetName) {{
            el.classList.add('highlighted');
            if (scrollIntoViewFlag) {{
              el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
          }} else {{
            el.classList.remove('highlighted');
          }}
        }});

        applyPlotStyles();
      }}

      function clearHighlightOnly() {{
        if (!activeHighlightName) return;
        activeHighlightName = null;
        selectedDetail.style.display = 'none';
        itemEls.forEach((el) => {{
          el.classList.remove('highlighted');
        }});
        applyPlotStyles();
      }}

      function resetAll() {{
        activeHighlightName = null;
        hiddenSet.clear();
        selectedDetail.style.display = 'none';
        itemEls.forEach((el) => {{
          el.classList.remove('highlighted');
          el.classList.remove('hidden-trace');
        }});
        applyPlotStyles();
      }}

      clearBtn.addEventListener('click', clearHighlightOnly);
      selectedDetail.addEventListener('click', clearHighlightOnly);
      resetBtn.addEventListener('click', resetAll);
      document.addEventListener('keydown', (e) => {{
        if (e.key === 'Escape') clearHighlightOnly();
      }});

      Plotly.newPlot(gd, figData.data, figData.layout, {{responsive: true, displaylogo: false}}).then(() => {{
        gd.on('plotly_hover', function() {{
          gd.classList.add('pointer-cursor');
        }});
        gd.on('plotly_unhover', function() {{
          gd.classList.remove('pointer-cursor');
        }});

        gd.on('plotly_click', function(ev) {{
          if (ev && ev.points && ev.points.length > 0) {{
            const pt = ev.points[0];
            const tName = pt.data.name || pt.data.legendgroup;
            if (tName) {{
              justClickedTrace = true;
              setTimeout(() => {{ justClickedTrace = false; }}, 80);
              if (activeHighlightName === tName) {{
                clearHighlightOnly();
              }} else {{
                selectAndHighlight(tName, true);
              }}
            }}
          }}
        }});

        gd.addEventListener('click', function() {{
          setTimeout(() => {{
            if (!justClickedTrace && activeHighlightName) {{
              clearHighlightOnly();
            }}
          }}, 20);
        }});
      }});
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=405)


def inject_sticky_lot_header_script():
    js_code = """
    <script>
    (function() {
      const parentDoc = window.parent.document;
      const parentWin = window.parent;

      if (parentWin.__odcsStickyCleanup) {
        parentWin.__odcsStickyCleanup();
      }

      function updateStickyHeader() {
        const marker = parentDoc.querySelector('.odcs-lot-sticky-marker');
        if (!marker) return;

        const headerRow = marker.closest('[data-testid="stHorizontalBlock"]');
        if (!headerRow || !headerRow.parentElement) return;

        let placeholder = parentDoc.getElementById('odcs-sticky-placeholder');
        if (!placeholder) {
          placeholder = parentDoc.createElement('div');
          placeholder.id = 'odcs-sticky-placeholder';
          placeholder.style.display = 'none';
          placeholder.style.width = '100%';
          headerRow.parentNode.insertBefore(placeholder, headerRow);
        }

        const stApp = parentDoc.querySelector('.stApp');
        const zoomVal = stApp ? (parseFloat(getComputedStyle(stApp).zoom) || 1.0) : 1.0;

        const parentRect = headerRow.parentElement.getBoundingClientRect();
        const isFixed = (headerRow.style.position === 'fixed');
        const triggerTop = isFixed
          ? placeholder.getBoundingClientRect().top
          : headerRow.getBoundingClientRect().top;

        if (parentRect.width === 0) {
          placeholder.style.display = 'none';
          headerRow.style.position = '';
          headerRow.style.top = '';
          headerRow.style.left = '';
          headerRow.style.width = '';
          headerRow.style.zIndex = '';
          headerRow.style.backgroundColor = '';
          headerRow.style.padding = '';
          headerRow.style.borderBottom = '';
          headerRow.style.boxShadow = '';
          return;
        }

        const stickyTopVisual = 48;

        if (triggerTop < stickyTopVisual) {
          placeholder.style.display = 'block';
          placeholder.style.height = headerRow.offsetHeight + 'px';

          headerRow.style.position = 'fixed';
          headerRow.style.top = (stickyTopVisual / zoomVal) + 'px';
          headerRow.style.left = (parentRect.left / zoomVal) + 'px';
          headerRow.style.width = (parentRect.width / zoomVal) + 'px';
          headerRow.style.boxSizing = 'border-box';
          headerRow.style.zIndex = '999';
          headerRow.style.backgroundColor = '#ffffff';
          headerRow.style.paddingTop = '10px';
          headerRow.style.paddingBottom = '10px';
          headerRow.style.borderBottom = '2px solid #cbd5e1';
          headerRow.style.boxShadow = '0 4px 12px rgba(15, 23, 42, 0.12)';
        } else {
          placeholder.style.display = 'none';
          headerRow.style.position = '';
          headerRow.style.top = '';
          headerRow.style.left = '';
          headerRow.style.width = '';
          headerRow.style.zIndex = '';
          headerRow.style.backgroundColor = '';
          headerRow.style.paddingTop = '';
          headerRow.style.paddingBottom = '';
          headerRow.style.borderBottom = '';
          headerRow.style.boxShadow = '';
        }
      }

      const onScrollOrResize = () => window.requestAnimationFrame(updateStickyHeader);
      parentWin.addEventListener('scroll', onScrollOrResize, true);
      parentWin.addEventListener('resize', onScrollOrResize);
      const timer = setInterval(updateStickyHeader, 300);

      parentWin.__odcsStickyCleanup = function() {
        parentWin.removeEventListener('scroll', onScrollOrResize, true);
        parentWin.removeEventListener('resize', onScrollOrResize);
        clearInterval(timer);
      };

      updateStickyHeader();
    })();
    </script>
    """
    components.html(js_code, height=0)


def _add_peak_span_arrow(fig: go.Figure, p_idx: int, y_base: float, y_peak: float, peak_val: float, color: str):
    fig.add_trace(go.Scatter(
        x=[p_idx, p_idx],
        y=[y_base, y_peak],
        mode="lines+markers",
        line=dict(color=color, width=1.8),
        marker=dict(
            symbol=["triangle-down", "diamond"],
            size=[7, 8],
            color=color,
        ),
        hovertemplate=f"+{peak_val:.2f} mNm (Peak: {y_peak:.2f} / Base: {y_base:.2f})<extra></extra>",
        showlegend=False,
    ))


def build_hdd_card_chart(
    parsed_data: dict,
    base_lot: str,
    hdd: str,
    trials: list,
    target_stages: list = None,
    show_baseline: bool = True,
    override_ref_key: tuple = None,
    x_upper: int = 2450,
    y_upper: float = 56.0,
):
    if target_stages is None:
        target_stages = ["Pre-HT", "HT"]

    fig = go.Figure()

    ht_pair_palette = {
        1: "darkorange",
        2: "#e11d48",
        3: "#7c3aed",
    }
    pre_pair_palette = {
        1: "royalblue",
        2: "#0284c7",
        3: "#0d9488",
    }

    override_ref_trq = None
    override_label = ""
    if override_ref_key is not None and override_ref_key in parsed_data:
        override_ref_trq = parsed_data[override_ref_key]["torque"]
        override_label = f"Ref: {override_ref_key[0]} Pre-{override_ref_key[3]}"

    recalc_results = {}
    sorted_trials = sorted(int(t) for t in trials)
    first_trial = sorted_trials[0] if sorted_trials else 1

    for tr in sorted_trials:
        opac = 1.0 if tr == 1 else 0.85
        tr_suffix = f"-{tr}" if (len(sorted_trials) > 1 or tr != 1) else ""

        if "Pre-HT" in target_stages and "HT" not in target_stages and (base_lot, hdd, "Pre-HT", tr) in parsed_data:
            d = parsed_data[(base_lot, hdd, "Pre-HT", tr)]
            an = (
                analyze_stiction(d["torque"], d["ul_error"], override_ref_trq)
                if override_ref_trq is not None
                else d.get("analysis")
            )
            if an:
                recalc_results[("Pre-HT", tr)] = an
            n_pts = len(d["torque"])
            step = max(1, n_pts // 400)
            idx_x = np.arange(0, n_pts, step)

            pre_col = "crimson" if d["ul_error"] else pre_pair_palette.get(tr, "royalblue")
            fig.add_trace(go.Scatter(
                x=idx_x, y=d["torque"][idx_x], mode="lines",
                name=f"Pre{tr_suffix} ({an['stiction_peak']:.2f})" if an else f"Pre-HT{tr_suffix}",
                line=dict(color=pre_col, width=2.1 if tr == 1 else 1.5, dash="solid"),
                opacity=opac,
            ))
            if (show_baseline or override_ref_trq is not None) and an and "baseline_curve" in an:
                fig.add_trace(go.Scatter(
                    x=idx_x, y=an["baseline_curve"][idx_x], mode="lines",
                    name=override_label if override_label else f"基準{tr_suffix}",
                    line=dict(color=pre_col, width=1.6, dash="dot"),
                    opacity=0.85,
                    showlegend=bool(tr == first_trial),
                ))

            if an and an.get("peak_idx") is not None and an.get("stiction_peak", 0.0) > 0:
                p_idx = an["peak_idx"]
                y_peak = float(d["torque"][p_idx])
                y_base = float(an["baseline_curve"][p_idx]) if "baseline_curve" in an else (y_peak - an["stiction_peak"])
                _add_peak_span_arrow(fig, p_idx, y_base, y_peak, an["stiction_peak"], color="blue")

        if "HT" in target_stages and (base_lot, hdd, "HT", tr) in parsed_data:
            d = parsed_data[(base_lot, hdd, "HT", tr)]
            an = (
                analyze_stiction(d["torque"], d["ul_error"], override_ref_trq)
                if override_ref_trq is not None
                else d.get("analysis", {})
            )
            recalc_results[("HT", tr)] = an
            pk_val = an.get("stiction_peak", 0.0)
            n_pts = len(d["torque"])
            step = max(1, n_pts // 400)
            idx_x = np.arange(0, n_pts, step)

            pair_col = "crimson" if d["ul_error"] else ht_pair_palette.get(tr, "#d97706")

            fig.add_trace(go.Scatter(
                x=idx_x, y=d["torque"][idx_x], mode="lines",
                name=f"HT{tr_suffix} ({pk_val:.2f})",
                line=dict(color=pair_col, width=2.2 if tr == 1 else 1.6, dash="solid"),
                opacity=opac,
            ))
            if (show_baseline or override_ref_trq is not None) and "baseline_curve" in an:
                fig.add_trace(go.Scatter(
                    x=idx_x, y=an["baseline_curve"][idx_x], mode="lines",
                    name=override_label if override_label else f"Pre-HT{tr_suffix}",
                    line=dict(color=pair_col, width=1.6, dash="dot"),
                    opacity=0.85,
                    showlegend=bool(tr == first_trial),
                ))
            if an.get("peak_idx") is not None and pk_val > 0:
                p_idx = an["peak_idx"]
                y_peak = float(d["torque"][p_idx])
                y_base = float(an["baseline_curve"][p_idx]) if "baseline_curve" in an else (y_peak - pk_val)
                _add_peak_span_arrow(fig, p_idx, y_base, y_peak, pk_val, color="red")

    max_tick = int(np.ceil(x_upper / 500.0)) * 500
    tick_vals = list(range(0, max_tick + 1, 500))
    tick_texts = [f"{v / 1000.0:g}" for v in tick_vals]

    fig.update_layout(
        height=222,
        margin=dict(l=26, r=8, t=4, b=20),
        xaxis=dict(
            title=None,
            range=[-50, x_upper],
            tickvals=tick_vals,
            ticktext=tick_texts,
            tickfont=dict(size=9),
            showgrid=True,
            gridcolor="#eeeeee",
        ),
        yaxis=dict(title=None, range=[0, y_upper], tickfont=dict(size=9), showgrid=True, gridcolor="#eeeeee"),
        legend=dict(
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.72)", bordercolor="rgba(200, 200, 200, 0.5)",
            borderwidth=0.5, font=dict(size=8.5), itemwidth=30,
        ),
        plot_bgcolor="#FFFFF5",
        paper_bgcolor="white",
    )
    return fig, recalc_results


# ==========================================
# 6. サマリー一覧表生成（横スクロール対応＆見切れ防止）
# ==========================================
def build_lot_summary_table(df_view: pd.DataFrame, lot: str, selected_trials: list):
    lot_df = df_view[df_view["β+Lot / 条件"] == lot]
    hdds = sorted(lot_df["HDD No."].unique())

    existing_trials = {int(t) for t in lot_df["回数"].unique()}
    active_trials = sorted([int(t) for t in selected_trials if int(t) in existing_trials])

    judge_col_name = tr_txt("tbl_judge")
    rows = []
    for i, hdd in enumerate(hdds, start=1):
        sub = (
            lot_df[lot_df["HDD No."] == hdd]
            .drop_duplicates(subset=["熱処理", "回数"])
            .sort_values(["熱処理", "回数"], ascending=[False, True])
        )
        has_pre = "Pre-HT" in sub["熱処理"].values
        has_ht = "HT" in sub["熱処理"].values
        show_stage_lbl = (has_pre and has_ht)

        trial_vals = {t: [] for t in active_trials}
        for _, r in sub.iterrows():
            tr = int(r["回数"])
            if tr in trial_vals:
                pk_val = r["粘着ピーク [mN・m]"]
                stg = f"{r['熱処理']}: " if show_stage_lbl else ""
                trial_vals[tr].append(f"{stg}{pk_val:.2f}")

        j_codes = set(sub["judge_code"].values)
        if any(c in ("NG", "NG_UL") for c in j_codes):
            ul_note = " (UL)" if "NG_UL" in j_codes else ""
            judge_str = f"🔴 {tr_txt('judge_disp_ng')}{ul_note}"
            judge_rank = 2
        elif "OUT_OF_CONTROL" in j_codes:
            judge_str = f"🟡 {tr_txt('judge_disp_warn')}"
            judge_rank = 1
        else:
            judge_str = f"🟢 {tr_txt('judge_disp_ok')}"
            judge_rank = 0

        row_dict = {
            "No.": f"N={i}",
            "HDD No.": hdd,
        }
        for t in active_trials:
            col_lbl = tr_txt("tbl_trial_col", t=t)
            row_dict[col_lbl] = " / ".join(trial_vals[t]) if trial_vals[t] else "-"
        row_dict[judge_col_name] = judge_str
        row_dict["_rank"] = judge_rank
        rows.append(row_dict)

    return pd.DataFrame(rows)


def render_scrollable_summary_table(df_sum: pd.DataFrame):
    if df_sum.empty:
        return
    cols = [c for c in df_sum.columns if c != "_rank"]
    judge_col_name = tr_txt("tbl_judge")
    thead = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    tbody_rows = []
    for _, row in df_sum.iterrows():
        rank = int(row.get("_rank", 0))
        tds = []
        for c in cols:
            val_str = html.escape(str(row[c]))
            if c == judge_col_name and rank == 2:
                tds.append(f'<td style="color:#dc2626;font-weight:700;">{val_str}</td>')
            elif c == judge_col_name and rank == 1:
                tds.append(f'<td style="color:#d97706;font-weight:700;">{val_str}</td>')
            elif c == judge_col_name and rank == 0:
                tds.append(f'<td style="color:#16a34a;font-weight:700;">{val_str}</td>')
            else:
                tds.append(f"<td>{val_str}</td>")
        tbody_rows.append(f"<tr>{''.join(tds)}</tr>")

    table_html = f"""
    <div class="odcs-scroll-table-wrap">
      <table class="odcs-scroll-table">
        <thead><tr>{thead}</tr></thead>
        <tbody>{''.join(tbody_rows)}</tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def format_yymmdd(d_str: str) -> str:
    if len(d_str) == 6 and d_str.isdigit():
        return f"{d_str} (20{d_str[:2]}/{d_str[2:4]}/{d_str[4:]})"
    return d_str


def checkbox_dropdown_filter(
    label: str,
    options: list,
    default_selected: list,
    key_prefix: str,
    format_func=str,
    max_selections: int = None,
):
    st.markdown(f"**{label}**")
    if not options:
        st.caption(tr_txt("no_options"))
        return []

    sig_key = f"{key_prefix}_sig"
    sel_key = f"{key_prefix}_selected"
    sa_key = f"{key_prefix}_sa"
    opt_sig = tuple(options)

    def _sync_children_from_selected(selected_list):
        selected_set = set(selected_list)
        for i, o in enumerate(options):
            st.session_state[f"{key_prefix}_cb_{i}"] = (o in selected_set)
        if max_selections is None:
            st.session_state[sa_key] = (len(selected_set) == len(options))

    def _on_select_all_change():
        target_state = bool(st.session_state.get(sa_key, False))
        new_list = list(options) if target_state else []
        st.session_state[sel_key] = new_list
        for i in range(len(options)):
            st.session_state[f"{key_prefix}_cb_{i}"] = target_state

    def _on_item_change():
        picked = [o for i, o in enumerate(options) if bool(st.session_state.get(f"{key_prefix}_cb_{i}", False))]
        if max_selections and len(picked) > max_selections:
            picked = picked[:max_selections]
            for i, o in enumerate(options):
                st.session_state[f"{key_prefix}_cb_{i}"] = (o in picked)
        st.session_state[sel_key] = picked
        if max_selections is None:
            st.session_state[sa_key] = (len(picked) == len(options))

    if sig_key not in st.session_state or st.session_state[sig_key] != opt_sig:
        st.session_state[sig_key] = opt_sig
        init_vals = [x for x in default_selected if x in options]
        if max_selections and len(init_vals) > max_selections:
            init_vals = init_vals[:max_selections]
        st.session_state[sel_key] = init_vals
        _sync_children_from_selected(init_vals)

    cur_selected = [x for x in st.session_state.get(sel_key, []) if x in options]

    if max_selections is None and len(cur_selected) == len(options):
        if not all(st.session_state.get(f"{key_prefix}_cb_{i}", False) for i in range(len(options))):
            _sync_children_from_selected(options)

    if len(cur_selected) == len(options):
        btn_text = tr_txt("all_selected", n=len(options))
    elif len(cur_selected) == 0:
        btn_text = tr_txt("none_selected")
    elif len(cur_selected) <= 2:
        btn_text = ", ".join(format_func(x) for x in cur_selected)
    else:
        btn_text = tr_txt("more_selected", first=format_func(cur_selected[0]), rest=len(cur_selected) - 1)

    with st.popover(btn_text, use_container_width=True):
        q = st.text_input(
            "Search",
            key=f"{key_prefix}_search",
            placeholder=tr_txt("search_placeholder"),
            label_visibility="collapsed",
        )

        if max_selections is None:
            all_checked = (len(cur_selected) == len(options))
            st.session_state[sa_key] = all_checked
            c_sa_lbl, c_sa_chk = st.columns([0.82, 0.18])
            c_sa_lbl.markdown("**Select all**")
            c_sa_chk.checkbox(
                "Select all",
                value=all_checked,
                key=sa_key,
                on_change=_on_select_all_change,
                label_visibility="collapsed",
            )
            st.divider()

        for idx, opt in enumerate(options):
            disp = format_func(opt)
            cb_key = f"{key_prefix}_cb_{idx}"
            is_checked = (opt in cur_selected)
            st.session_state[cb_key] = is_checked

            if q and q.lower() not in disp.lower():
                continue

            c_item, c_box = st.columns([0.82, 0.18])
            c_item.markdown(disp)
            c_box.checkbox(
                disp,
                value=is_checked,
                key=cb_key,
                on_change=_on_item_change,
                label_visibility="collapsed",
            )

    return st.session_state[sel_key]


# ==========================================
# 7. Streamlit メインUI（初期化 & サイドバー）
# ==========================================
default_local_dir = r"C:\Users\OGAWA\Desktop\Stickness Peak Test"
default_src_idx = 0 if not os.path.isdir(default_local_dir) else 1

if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = LANG_OPTIONS[0]
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0
if "gdrive_folder_input" not in st.session_state:
    st.session_state.gdrive_folder_input = DEFAULT_GDRIVE_FOLDER_ID
if "target_dir" not in st.session_state:
    st.session_state.target_dir = default_local_dir
if "input_mode_idx" not in st.session_state:
    st.session_state.input_mode_idx = default_src_idx
if "baseline_mode_idx" not in st.session_state:
    st.session_state.baseline_mode_idx = 0
if "custom_ref_lot" not in st.session_state:
    st.session_state.custom_ref_lot = None
if "pre_ht_eval_lots" not in st.session_state:
    st.session_state.pre_ht_eval_lots = ["未処理"]
if "lot_ref_overrides" not in st.session_state:
    st.session_state.lot_ref_overrides = {}

models_data = {}
available_models = []
gdrive_err_msg = None

if st.session_state.input_mode_idx == 0:
    models_data, available_models, gdrive_err_msg = load_raw_dataset_from_gdrive_public(
        st.session_state.gdrive_folder_input,
        st.session_state.refresh_token,
    )
elif st.session_state.input_mode_idx == 1:
    if os.path.isdir(st.session_state.target_dir):
        models_data, available_models = load_raw_dataset_from_folder(
            st.session_state.target_dir,
            st.session_state.refresh_token,
        )
else:
    if "uploaded_tuples" in st.session_state and st.session_state.uploaded_tuples:
        models_data, available_models = load_raw_dataset_from_uploads(st.session_state.uploaded_tuples)

# 品番リストの変更検知＆初期選択（最新測定日優先 / 同率時は M040-000013-P2BKKI 優先）
models_sig = tuple(available_models)
if st.session_state.get("models_sig") != models_sig:
    st.session_state.models_sig = models_sig
    st.session_state.selected_model = select_default_model(models_data, available_models)

with st.sidebar:
    if st.button(tr_txt("btn_refresh"), use_container_width=True, type="primary"):
        st.session_state.refresh_token += 1
        st.rerun()

    # 更新ボタンのすぐ下に「品番選択（単一選択・拡張自動対応）」を配置
    if available_models:
        cur_model = st.session_state.get("selected_model", available_models[0])
        if cur_model not in available_models:
            cur_model = select_default_model(models_data, available_models)
            st.session_state.selected_model = cur_model

        chosen_model = st.selectbox(
            tr_txt("model_select_label"),
            options=available_models,
            index=available_models.index(cur_model),
            key="model_selector_box",
        )
        if chosen_model != cur_model:
            st.session_state.selected_model = chosen_model
            st.session_state.lot_ref_overrides = {}
            st.rerun()
    else:
        chosen_model = ""

raw_parsed_data = models_data.get(chosen_model, {}) if chosen_model else {}

all_base_lots = sorted({k[0] for k in raw_parsed_data.keys()}) if raw_parsed_data else []
valid_pre_eval = [l for l in st.session_state.pre_ht_eval_lots if l in all_base_lots]

b_mode = ["auto", "custom", "self"][st.session_state.baseline_mode_idx]
parsed_data, df_res = run_batch_analysis(
    raw_parsed_data,
    b_mode,
    st.session_state.custom_ref_lot,
    valid_pre_eval,
    lot_ref_overrides=st.session_state.lot_ref_overrides,
)

with st.sidebar:
    st.divider()

    st.header(tr_txt("sidebar_filter_header"))
    if not df_res.empty:
        all_lots = sorted(df_res["β+Lot / 条件"].unique())
        recent_3_lots = get_recent_lots(df_res, n=3)
        all_trials = sorted(int(x) for x in df_res["回数"].unique())
        all_dates = sorted(df_res["測定日"].unique(), reverse=True)
        all_hdds = sorted(df_res["HDD No."].unique())

        filter_ns = f"m_{chosen_model}"
        selected_lots = checkbox_dropdown_filter(
            tr_txt("f_lot"),
            options=all_lots,
            default_selected=recent_3_lots if recent_3_lots else [all_lots[0]],
            key_prefix=f"{filter_ns}_main_lot",
            max_selections=5,
        )
        selected_trials = checkbox_dropdown_filter(
            tr_txt("f_trial"),
            options=all_trials,
            default_selected=all_trials,
            key_prefix=f"{filter_ns}_main_trial",
            format_func=lambda x: tr_txt("trial_fmt", x=x),
        )
        selected_dates = checkbox_dropdown_filter(
            tr_txt("f_date"),
            options=all_dates,
            default_selected=all_dates,
            key_prefix=f"{filter_ns}_main_date",
            format_func=format_yymmdd,
        )
        avail_main_hdds = sorted(df_res[df_res["β+Lot / 条件"].isin(selected_lots)]["HDD No."].unique()) if selected_lots else all_hdds
        selected_hdds = checkbox_dropdown_filter(
            tr_txt("f_hdd"),
            options=avail_main_hdds,
            default_selected=avail_main_hdds,
            key_prefix=f"{filter_ns}_main_hdd",
        )
    else:
        selected_lots, selected_trials, selected_dates, selected_hdds = [], [], [], []

    st.divider()

    st.subheader(tr_txt("sidebar_display_header"))
    only_ng_rows = st.checkbox(tr_txt("only_ng"), value=False)
    global_show_base = st.checkbox(tr_txt("show_base"), value=True)

    auto_scale = st.checkbox(tr_txt("auto_scale"), value=True)
    manual_y_val = None
    manual_x_val = None
    if not auto_scale:
        if "scale_y_sync" not in st.session_state:
            st.session_state.scale_y_sync = 55.0
        if "scale_x_sync" not in st.session_state:
            st.session_state.scale_x_sync = 2500

        def _on_y_slider():
            st.session_state.scale_y_num = st.session_state.scale_y_slider
            st.session_state.scale_y_sync = st.session_state.scale_y_slider

        def _on_y_num():
            st.session_state.scale_y_slider = st.session_state.scale_y_num
            st.session_state.scale_y_sync = st.session_state.scale_y_num

        def _on_x_slider():
            st.session_state.scale_x_num = st.session_state.scale_x_slider
            st.session_state.scale_x_sync = st.session_state.scale_x_slider

        def _on_x_num():
            st.session_state.scale_x_slider = st.session_state.scale_x_num
            st.session_state.scale_x_sync = st.session_state.scale_x_num

        st.caption(tr_txt("y_scale_cap"))
        cy1, cy2 = st.columns([0.62, 0.38])
        cy1.slider(
            "Y slider",
            min_value=5.0,
            max_value=100.0,
            value=float(st.session_state.scale_y_sync),
            step=1.0,
            key="scale_y_slider",
            on_change=_on_y_slider,
            label_visibility="collapsed",
        )
        cy2.number_input(
            "Y num",
            min_value=1.0,
            max_value=200.0,
            value=float(st.session_state.scale_y_sync),
            step=1.0,
            key="scale_y_num",
            on_change=_on_y_num,
            label_visibility="collapsed",
        )
        manual_y_val = float(st.session_state.scale_y_sync)

        st.caption(tr_txt("x_scale_cap"))
        cx1, cx2 = st.columns([0.62, 0.38])
        cx1.slider(
            "X slider",
            min_value=500,
            max_value=5000,
            value=int(st.session_state.scale_x_sync),
            step=100,
            key="scale_x_slider",
            on_change=_on_x_slider,
            label_visibility="collapsed",
        )
        cx2.number_input(
            "X num",
            min_value=200,
            max_value=10000,
            value=int(st.session_state.scale_x_sync),
            step=100,
            key="scale_x_num",
            on_change=_on_x_num,
            label_visibility="collapsed",
        )
        manual_x_val = int(st.session_state.scale_x_sync)

    st.divider()

    with st.expander(tr_txt("exp_data_settings"), expanded=bool(gdrive_err_msg)):
        src_labels = [tr_txt("src_gdrive"), tr_txt("src_folder"), tr_txt("src_upload")]
        chosen_src = st.radio(
            tr_txt("input_src"),
            options=src_labels,
            index=st.session_state.input_mode_idx,
        )
        new_src_idx = src_labels.index(chosen_src)
        if new_src_idx != st.session_state.input_mode_idx:
            st.session_state.input_mode_idx = new_src_idx
            st.rerun()

        if st.session_state.input_mode_idx == 0:
            st.text_input(tr_txt("gdrive_folder_id"), key="gdrive_folder_input")
            if gdrive_err_msg:
                st.error(tr_txt("gdrive_err", err=gdrive_err_msg))
        elif st.session_state.input_mode_idx == 1:
            st.text_input(tr_txt("folder_path"), key="target_dir")
            if not os.path.isdir(st.session_state.target_dir):
                st.error(tr_txt("folder_not_found"))
        else:
            uploaded_files = st.file_uploader(
                tr_txt("uploader_label"),
                type=["xls", "xlsx", "zip"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                st.session_state.uploaded_tuples = tuple((uf.name, uf.getvalue()) for uf in uploaded_files)
                st.rerun()

        base_labels = [tr_txt("base_auto"), tr_txt("base_custom"), tr_txt("base_self")]
        chosen_base = st.radio(
            tr_txt("base_mode_label"),
            options=base_labels,
            index=st.session_state.baseline_mode_idx,
        )
        new_base_idx = base_labels.index(chosen_base)
        if new_base_idx != st.session_state.baseline_mode_idx:
            st.session_state.baseline_mode_idx = new_base_idx
            st.rerun()

        if st.session_state.baseline_mode_idx == 1 and all_base_lots:
            default_idx = next((i for i, l in enumerate(all_base_lots) if l == "β+量産品(TKR)" or l.startswith("KB-")), 0)
            st.selectbox(tr_txt("custom_ref_sel"), all_base_lots, index=default_idx, key="custom_ref_lot")

        if all_base_lots:
            st.multiselect(
                tr_txt("pre_ht_eval_sel"),
                all_base_lots,
                default=["未処理"] if "未処理" in all_base_lots else [],
                key="pre_ht_eval_lots",
            )

# ==========================================
# 8. メイン画面エリア（右上に言語選択ドロップダウン配置）
# ==========================================
title_col, lang_col = st.columns([0.82, 0.18])
with title_col:
    model_badge_str = f" — `{chosen_model}`" if chosen_model else ""
    st.title(f"{tr_txt('app_title')}{model_badge_str}")
with lang_col:
    st.selectbox(
        tr_txt("lang_label"),
        options=LANG_OPTIONS,
        key="ui_lang",
    )

if df_res.empty:
    st.info(tr_txt("info_no_data"))
else:
    pre_ht_base_lots = sorted({k[0] for k in raw_parsed_data.keys() if k[2] == "Pre-HT"})

    df_view = df_res[
        df_res["β+Lot / 条件"].isin(selected_lots)
        & df_res["回数"].isin(selected_trials)
        & df_res["測定日"].isin(selected_dates)
        & df_res["HDD No."].isin(selected_hdds)
    ]

    cnt_total = len(df_view)
    cnt_ok = int((df_view["judge_code"] == "OK").sum())
    cnt_warn = int((df_view["judge_code"] == "OUT_OF_CONTROL").sum())
    cnt_ng = int(df_view["judge_code"].isin(["NG", "NG_UL"]).sum())

    m1, m2, m3, m4, m5 = st.columns([0.9, 1.0, 1.15, 1.05, 1.4])
    m1.metric(tr_txt("metric_total"), tr_txt("unit_items", n=cnt_total))
    m2.metric(tr_txt("metric_ok"), tr_txt("unit_items", n=cnt_ok))
    m3.metric(tr_txt("metric_warn"), tr_txt("unit_items", n=cnt_warn))
    m4.metric(tr_txt("metric_ng"), tr_txt("unit_items", n=cnt_ng))

    export_df = df_view.drop(columns=["base_lot", "mtime", "judge_code"], errors="ignore")
    csv_fname = f"ODCS_Stiction_{chosen_model}.csv" if chosen_model else "ODCS_Stiction_Judgement.csv"
    m5.download_button(
        tr_txt("btn_csv"),
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=csv_fname,
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader(tr_txt("sec_summary"))
    if not selected_lots or not selected_trials or not selected_dates or not selected_hdds:
        st.warning(tr_txt("warn_select_filter"))
    elif df_view.empty:
        st.warning(tr_txt("warn_no_match"))
    else:
        sum_cols = st.columns(len(selected_lots))
        for idx_lot, lot in enumerate(selected_lots):
            with sum_cols[idx_lot]:
                st.markdown(f"**🏷️ {lot}**")
                df_sum = build_lot_summary_table(df_view, lot, selected_trials)
                render_scrollable_summary_table(df_sum)

        if auto_scale:
            global_x_max = 2000
            global_y_max = 10.0
            for lot in selected_lots:
                t_stages = ["Pre-HT"] if lot == "未処理 (Pre-HT)" else (["HT"] if lot == "未処理 (HT)" else ["Pre-HT", "HT"])
                sub_lot = df_view[df_view["β+Lot / 条件"] == lot]
                for _, r in sub_lot.drop_duplicates(subset=["base_lot", "HDD No."]).iterrows():
                    b_lot, h_no = r["base_lot"], r["HDD No."]
                    for tr in selected_trials:
                        for stg in t_stages:
                            if (b_lot, h_no, stg, int(tr)) in parsed_data:
                                trq_arr = parsed_data[(b_lot, h_no, stg, int(tr))]["torque"]
                                if len(trq_arr) > 0:
                                    global_x_max = max(global_x_max, len(trq_arr))
                                    global_y_max = max(global_y_max, float(np.max(trq_arr)))

            global_x_upper = int(np.ceil(global_x_max * 1.03))
            global_y_upper = float(np.ceil((global_y_max * 1.08) / 5.0) * 5.0)
        else:
            global_x_upper = manual_x_val
            global_y_upper = manual_y_val

        st.divider()

        st.subheader(tr_txt("sec_overlay"))

        ov_col_chart, ov_col_filter = st.columns([0.75, 0.25])

        with ov_col_filter.container(border=True):
            st.markdown(tr_txt("ov_filter_title"))

            ov_lots = checkbox_dropdown_filter(
                tr_txt("ov_lot"),
                options=list(selected_lots),
                default_selected=list(selected_lots),
                key_prefix=f"m_{chosen_model}_ov_lot",
            )
            avail_ov_hdds = sorted(df_view[df_view["β+Lot / 条件"].isin(ov_lots)]["HDD No."].unique())
            ov_hdds = checkbox_dropdown_filter(
                tr_txt("ov_hdd"),
                options=avail_ov_hdds,
                default_selected=avail_ov_hdds,
                key_prefix=f"m_{chosen_model}_ov_hdd",
            )
            ref_mode_labels = [tr_txt("ov_ref_fit"), tr_txt("ov_ref_raw"), tr_txt("ov_ref_none")]
            ov_ref_choice = st.radio(
                tr_txt("ov_ref_mode"),
                options=ref_mode_labels,
                index=0,
                key=f"m_{chosen_model}_ov_filter_ref_mode",
            )
            ov_ref_mode = ["fit", "raw", "none"][ref_mode_labels.index(ov_ref_choice)]

            ov_trials = checkbox_dropdown_filter(
                tr_txt("ov_tr"),
                options=[int(x) for x in selected_trials],
                default_selected=[int(x) for x in selected_trials],
                key_prefix=f"m_{chosen_model}_ov_tr",
                format_func=lambda x: tr_txt("trial_fmt", x=x),
            )
            judge_opts = [
                tr_txt("judge_all"),
                tr_txt("judge_ok"),
                tr_txt("judge_warn"),
                tr_txt("judge_ng"),
                tr_txt("judge_ng_warn"),
            ]
            ov_judge = st.selectbox(
                tr_txt("ov_judge"),
                options=judge_opts,
                index=4 if only_ng_rows else 0,
                key=f"m_{chosen_model}_ov_filter_judge",
            )

        df_ov = df_view[
            df_view["β+Lot / 条件"].isin(ov_lots)
            & df_view["HDD No."].isin(ov_hdds)
        ]
        if ov_judge == tr_txt("judge_ok"):
            df_ov = df_ov[df_ov["judge_code"] == "OK"]
        elif ov_judge == tr_txt("judge_warn"):
            df_ov = df_ov[df_ov["judge_code"] == "OUT_OF_CONTROL"]
        elif ov_judge == tr_txt("judge_ng"):
            df_ov = df_ov[df_ov["judge_code"].isin(["NG", "NG_UL"])]
        elif ov_judge == tr_txt("judge_ng_warn") or only_ng_rows:
            df_ov = df_ov[df_ov["judge_code"] != "OK"]

        with ov_col_chart.container(border=True):
            fig_overlay, overlay_displayed_cards = build_combined_overlay_chart(
                parsed_data,
                df_ov,
                ov_lots,
                ov_trials,
                ov_ref_mode=ov_ref_mode,
                x_upper=global_x_upper,
                y_upper=global_y_upper,
            )
            render_interactive_overlay_chart(fig_overlay)

        st.divider()

        tab_matrix, tab_detail = st.tabs([tr_txt("tab_cards"), tr_txt("tab_table")])

        with tab_matrix:
            master_hdds = sorted(df_view["HDD No."].unique())

            lot_ref_codes = ["__AUTO__", "__SELF__"] + pre_ht_base_lots

            header_cols = st.columns(len(selected_lots))
            for idx_lot, lot in enumerate(selected_lots):
                lot_df = df_view[df_view["β+Lot / 条件"] == lot]
                ng_cnt = int(lot_df["judge_code"].isin(["NG", "NG_UL"]).sum())
                warn_cnt = int((lot_df["judge_code"] == "OUT_OF_CONTROL").sum())
                max_pk = lot_df["粘着ピーク [mN・m]"].max() if not lot_df.empty else 0.0
                ref_used = lot_df["基準ベースライン"].iloc[0] if not lot_df.empty else "-"
                lot_hdd_cnt = lot_df["HDD No."].nunique()
                lot_esc = html.escape(str(lot))
                marker_cls = "odcs-lot-sticky-marker " if idx_lot == 0 else ""

                with header_cols[idx_lot].container(border=True):
                    top_html = f"""
                    <div class="{marker_cls}odcs-lot-header-top" title="{lot_esc}">
                      <div class="odcs-lot-header-title">🏷️ {lot_esc}</div>
                      <div class="odcs-lot-header-sub">
                        N={lot_hdd_cnt} ｜ {tr_txt('hdr_max')}: <b>{max_pk:.2f}</b> ｜ {tr_txt('hdr_warn')}: <b>{warn_cnt}</b> ｜ {tr_txt('hdr_ng')}: <b>{ng_cnt}</b>
                      </div>
                    </div>
                    """
                    st.markdown(top_html, unsafe_allow_html=True)

                    c_ref_lbl, c_ref_sel = st.columns([0.22, 0.78])
                    c_ref_lbl.markdown(f'<div class="odcs-ref-label-inline">{tr_txt("hdr_base")}</div>', unsafe_allow_html=True)

                    cur_ov_code = st.session_state.lot_ref_overrides.get(lot, "__AUTO__")
                    if cur_ov_code not in lot_ref_codes:
                        cur_ov_code = "__AUTO__"

                    def _fmt_lot_ref(code, auto_str=ref_used):
                        if code == "__AUTO__":
                            return tr_txt("lot_ref_auto", auto_lbl=auto_str)
                        if code == "__SELF__":
                            return tr_txt("lot_ref_self")
                        return f"{code} Pre-HT"

                    chosen_code = c_ref_sel.selectbox(
                        f"Ref for {lot}",
                        options=lot_ref_codes,
                        index=lot_ref_codes.index(cur_ov_code),
                        format_func=_fmt_lot_ref,
                        key=f"m_{chosen_model}_lot_hdr_ref_sel_{lot}",
                        label_visibility="collapsed",
                    )
                    if chosen_code != cur_ov_code:
                        if chosen_code == "__AUTO__":
                            st.session_state.lot_ref_overrides.pop(lot, None)
                        else:
                            st.session_state.lot_ref_overrides[lot] = chosen_code
                        st.rerun()

            inject_sticky_lot_header_script()

            for row_idx, hdd in enumerate(master_hdds):
                if only_ng_rows and not (df_view[df_view["HDD No."] == hdd]["judge_code"] != "OK").any():
                    continue

                row_cols = st.columns(len(selected_lots))

                for idx_lot, lot in enumerate(selected_lots):
                    col_ui = row_cols[idx_lot]
                    hdd_df = (
                        df_view[(df_view["β+Lot / 条件"] == lot) & (df_view["HDD No."] == hdd)]
                        .drop_duplicates(subset=["熱処理", "回数"])
                        .sort_values(["熱処理", "回数"], ascending=[False, True])
                    )

                    if hdd_df.empty:
                        col_ui.markdown(
                            f'<div class="odcs-empty-card-box"><b>{html.escape(hdd)}</b><span>({tr_txt("no_data_card")})</span></div>',
                            unsafe_allow_html=True,
                        )
                        continue

                    base_lot = hdd_df["base_lot"].iloc[0]
                    dates = [d for d in hdd_df["測定日"].unique() if d != "-"]
                    date_str = f" ({dates[0]})" if dates else ""

                    if lot == "未処理 (Pre-HT)":
                        target_stages = ["Pre-HT"]
                    elif lot == "未処理 (HT)":
                        target_stages = ["HT"]
                    else:
                        target_stages = ["Pre-HT", "HT"]

                    same_hdd_pre_keys = sorted(
                        [k for k in parsed_data.keys() if k[1] == hdd and k[2] == "Pre-HT"],
                        key=lambda x: (0 if x[0] == base_lot else 1, x[0], x[3]),
                    )
                    auto_opt_lbl = tr_txt("ref_auto_opt")
                    ref_options = [auto_opt_lbl] + [
                        f"{k[0]} / Pre-HT (-{k[3]})" for k in same_hdd_pre_keys
                    ]
                    key_map = {f"{k[0]} / Pre-HT (-{k[3]})": k for k in same_hdd_pre_keys}

                    with col_ui.container(border=True):
                        c_title, c_btn = st.columns([0.82, 0.18])

                        with c_btn.popover("⚙️", use_container_width=True):
                            st.caption(tr_txt("pop_ref_cap", hdd=hdd))
                            selected_ref_label = st.selectbox(
                                "Ref select",
                                options=ref_options,
                                index=0,
                                key=f"m_{chosen_model}_ref_sel_{lot}_{hdd}_{row_idx}",
                                label_visibility="collapsed",
                            )
                            override_key = key_map.get(selected_ref_label)

                        fig, recalc_map = build_hdd_card_chart(
                            parsed_data,
                            base_lot,
                            hdd,
                            selected_trials,
                            target_stages=target_stages,
                            show_baseline=global_show_base,
                            override_ref_key=override_key,
                            x_upper=global_x_upper,
                            y_upper=global_y_upper,
                        )

                        has_pre_eval = "Pre-HT" in hdd_df["熱処理"].values
                        has_ht_eval = "HT" in hdd_df["熱処理"].values
                        show_stg_lbl = (has_pre_eval and has_ht_eval)

                        val_parts = []
                        j_codes_card = set()
                        for _, r in hdd_df.iterrows():
                            stg = r["熱処理"]
                            tr = int(r["回数"])
                            recalc_an = recalc_map.get((stg, tr))
                            pk_val = recalc_an["stiction_peak"] if recalc_an else r["粘着ピーク [mN・m]"]
                            j_code = recalc_an["judge"] if recalc_an else r["judge_code"]
                            j_codes_card.add(j_code)

                            stg_prefix = f"{stg}" if show_stg_lbl else ""
                            tr_prefix = f"-{tr}" if (len(selected_trials) > 1 or tr != 1) else ""
                            label = f"{stg_prefix}{tr_prefix}".strip("-")
                            val_parts.append(f"{stg_prefix}{tr_prefix}: {pk_val:.2f}" if label else f"{pk_val:.2f}")
                        val_str = " / ".join(val_parts) + " mNm"

                        if any(c in ("NG", "NG_UL") for c in j_codes_card):
                            ul_note = " (UL)" if "NG_UL" in j_codes_card else ""
                            badge_html = f'<span class="odcs-badge-ng">🔴 {html.escape(tr_txt("judge_disp_ng"))}{ul_note}</span>'
                        elif "OUT_OF_CONTROL" in j_codes_card:
                            badge_html = f'<span class="odcs-badge-warn">🟡 {html.escape(tr_txt("judge_disp_warn"))}</span>'
                        else:
                            badge_html = f'<span class="odcs-badge-ok">🟢 {html.escape(tr_txt("judge_disp_ok"))}</span>'

                        ov_badge_html = f'<span class="odcs-badge-ov">{html.escape(tr_txt("badge_in_ov"))}</span>' if (lot, hdd) in overlay_displayed_cards else ""

                        title_html = f"""
                        <div class="odcs-card-title-box" title="{html.escape(hdd)}{html.escape(date_str)} | {html.escape(val_str)}">
                          <div class="odcs-card-line1"><b>{html.escape(hdd)}</b>{html.escape(date_str)} {ov_badge_html}</div>
                          <div class="odcs-card-line2">{badge_html} <span class="odcs-val-pill">{html.escape(val_str)}</span></div>
                        </div>
                        """
                        c_title.markdown(title_html, unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True, key=f"m_{chosen_model}_chart_{lot}_{hdd}_{row_idx}")

        with tab_detail:
            st.dataframe(export_df, use_container_width=True, height=500)
