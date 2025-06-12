#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上海印章管理系统爬虫配置文件
Configuration file for Shanghai Seal Management System Scraper
"""

# 网站基本配置
BASE_URL = "https://gaj.sh.gov.cn/yzxt/"

# 超时配置 (秒)
DEFAULT_TIMEOUT = 30
PAGE_LOAD_TIMEOUT = 15
ELEMENT_WAIT_TIMEOUT = 10

# 查询延迟配置 (秒)
DEFAULT_QUERY_DELAY = 2.0
MIN_QUERY_DELAY = 1.0
MAX_QUERY_DELAY = 10.0

# 浏览器配置
BROWSER_CONFIG = {
    'headless': True,
    'window_size': (1920, 1080),
    'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'disable_images': True,  # 禁用图片加载以提高速度
    'disable_javascript': False,  # 某些功能可能需要JavaScript
}

# 页面元素选择器 (XPath/CSS)
# 注意：这些选择器需要根据实际网站结构进行调整
SELECTORS = {
    # 登录相关
    'enterprise_login_btn': "//button[contains(text(), '企业用户')]",
    'individual_login_btn': "//button[contains(text(), '非企业用户')]", 
    'username_input': "//input[@name='username'] | //input[@placeholder='用户名'] | //input[@id='username']",
    'password_input': "//input[@name='password'] | //input[@placeholder='密码'] | //input[@id='password']",
    'login_submit_btn': "//button[contains(text(), '登录')] | //input[@type='submit']",
    
    # 查询表单相关
    'company_name_input': [
        "//input[@name='companyName']",
        "//input[@placeholder='公司名称']",
        "//input[@placeholder='单位名称']",
        "//input[contains(@placeholder, '企业名称')]"
    ],
    'credit_code_input': [
        "//input[@name='creditCode']", 
        "//input[@placeholder='统一社会信用代码']",
        "//input[@placeholder='信用代码']",
        "//input[contains(@placeholder, '信用代码')]"
    ],
    'legal_person_input': [
        "//input[@name='legalPerson']",
        "//input[@placeholder='法人姓名']",
        "//input[@placeholder='法定代表人']"
    ],
    'legal_id_input': [
        "//input[@name='legalId']",
        "//input[@placeholder='身份证号']",
        "//input[@placeholder='法人身份证']"
    ],
    'search_btn': "//button[contains(text(), '查询')] | //button[contains(text(), '搜索')] | //input[@value='查询']",
    
    # 结果相关
    'results_table': "//table | //div[@class='table-responsive']//table",
    'result_rows': "//tr[position()>1] | //tbody//tr",
    'no_results_msg': "//div[contains(text(), '没有找到')] | //div[contains(text(), '无数据')] | //span[contains(text(), '暂无')]",
    
    # 分页相关
    'next_page_btn': "//a[contains(text(), '下一页')] | //button[contains(text(), '下一页')]",
    'page_numbers': "//a[contains(@class, 'page')] | //button[contains(@class, 'page')]"
}

# 重试配置
RETRY_CONFIG = {
    'max_retries': 3,
    'backoff_factor': 1,
    'status_forcelist': [429, 500, 502, 503, 504],
    'retry_on_timeout': True
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'shyz_scraper.log',
    'encoding': 'utf-8'
}

# 输出文件配置
OUTPUT_CONFIG = {
    'default_format': 'csv',
    'csv_encoding': 'utf-8-sig',  # 支持Excel打开中文
    'json_ensure_ascii': False,
    'json_indent': 2
}

# 错误处理配置
ERROR_CONFIG = {
    'save_failed_queries': True,
    'failed_queries_file': 'failed_queries.json',
    'save_screenshots_on_error': True,
    'screenshot_dir': 'error_screenshots'
}

# 印章信息字段映射
# 根据实际网站的表格结构调整
SEAL_INFO_FIELDS = {
    'seal_type': 0,          # 印章类型
    'seal_status': 1,        # 印章状态  
    'registration_date': 2,  # 登记日期
    'seal_code': 3,          # 印章编号
    'seal_material': 4,      # 印章材质
    'seal_size': 5,          # 印章尺寸
    'authorized_person': 6,  # 授权人
    'remarks': 7             # 备注
}

# 表格列标题映射（用于动态识别列）
COLUMN_MAPPING = {
    '印章类型': 'seal_type',
    '印章状态': 'seal_status', 
    '状态': 'seal_status',
    '登记日期': 'registration_date',
    '备案日期': 'registration_date',
    '印章编号': 'seal_code',
    '编号': 'seal_code',
    '印章材质': 'seal_material',
    '材质': 'seal_material',
    '印章尺寸': 'seal_size',
    '尺寸': 'seal_size',
    '授权人': 'authorized_person',
    '经办人': 'authorized_person',
    '备注': 'remarks',
    '说明': 'remarks'
}

# 状态值标准化映射
STATUS_MAPPING = {
    '正常': '正常',
    '有效': '正常', 
    '启用': '正常',
    '已注销': '注销',
    '注销': '注销',
    '失效': '注销',
    '停用': '停用',
    '暂停': '停用'
}

# 验证码处理配置
CAPTCHA_CONFIG = {
    'enable_auto_recognition': False,  # 是否启用自动识别验证码
    'manual_input_timeout': 30,       # 手动输入验证码的超时时间
    'max_captcha_attempts': 3         # 验证码最大尝试次数
} 