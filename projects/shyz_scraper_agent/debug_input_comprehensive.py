#!/usr/bin/env python3
"""
全面的输入调试脚本
专门测试各种输入方法，找出问题所在
"""

import time
import logging
from shyz_scraper_drission import SHYZScraperDrission, CompanyQuery

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_all_input_methods():
    """测试所有可能的输入方法"""
    print("=== 全面输入方法测试 ===")
    
    test_company = "上海遇圆实业有限公司"
    test_code = "91310000MAE97Y0K5Y"
    
    print(f"测试公司: {test_company}")
    print(f"测试代码: {test_code}")
    print()
    
    # 创建爬虫实例 - 显示浏览器便于观察
    scraper = SHYZScraperDrission(headless=False, timeout=30)
    
    try:
        print("1. 导航到查询页面...")
        success = scraper.navigate_to_query_page()
        
        if not success:
            print("❌ 无法进入查询页面")
            return
            
        print("✅ 成功进入查询页面")
        
        # 找到输入框
        company_input = scraper.page.ele("xpath://input[contains(@placeholder, '单位名称')]")
        code_input = scraper.page.ele("xpath://input[contains(@placeholder, '社会信用代码')]")
        
        if not company_input or not code_input:
            print("❌ 无法找到输入框")
            return
            
        print("✅ 找到两个输入框")
        
        # 定义所有测试方法
        input_methods = [
            {
                'name': '方法1: 简单input',
                'func': lambda inp, val: simple_input(scraper, inp, val)
            },
            {
                'name': '方法2: clear + input',
                'func': lambda inp, val: clear_and_input(scraper, inp, val)
            },
            {
                'name': '方法3: JavaScript setValue',
                'func': lambda inp, val: js_set_value(scraper, inp, val)
            },
            {
                'name': '方法4: JavaScript + 事件',
                'func': lambda inp, val: js_with_events(scraper, inp, val)
            },
            {
                'name': '方法5: 模拟键盘输入',
                'func': lambda inp, val: keyboard_input(scraper, inp, val)
            },
            {
                'name': '方法6: 剪贴板粘贴',
                'func': lambda inp, val: clipboard_paste(scraper, inp, val)
            },
            {
                'name': '方法7: 逐字符键盘',
                'func': lambda inp, val: char_by_char_keyboard(scraper, inp, val)
            }
        ]
        
        # 测试公司名称输入
        print("\n2. 测试公司名称输入...")
        company_success = False
        for method in input_methods:
            print(f"\n   尝试{method['name']}...")
            try:
                # 清空输入框
                clear_input_completely(scraper, company_input)
                time.sleep(0.5)
                
                # 尝试输入
                method['func'](company_input, test_company)
                time.sleep(1)
                
                # 验证结果
                current_value = company_input.value or company_input.text or ""
                current_value = current_value.strip()
                
                print(f"      结果: '{current_value}'")
                print(f"      期望: '{test_company}'")
                print(f"      匹配: {current_value == test_company}")
                
                if current_value == test_company:
                    print(f"   ✅ {method['name']} 成功！")
                    company_success = True
                    break
                else:
                    print(f"   ❌ {method['name']} 失败")
                    
            except Exception as e:
                print(f"   ❌ {method['name']} 出错: {e}")
        
        if not company_success:
            print("\n❌ 所有公司名称输入方法都失败")
            print("请手动输入公司名称...")
            input("输入完成后按回车继续...")
        
        # 测试信用代码输入
        print("\n3. 测试信用代码输入...")
        code_success = False
        for method in input_methods:
            print(f"\n   尝试{method['name']}...")
            try:
                # 清空输入框
                clear_input_completely(scraper, code_input)
                time.sleep(0.5)
                
                # 尝试输入
                method['func'](code_input, test_code)
                time.sleep(1)
                
                # 验证结果
                current_value = code_input.value or code_input.text or ""
                current_value = current_value.strip()
                
                print(f"      结果: '{current_value}'")
                print(f"      期望: '{test_code}'")
                print(f"      匹配: {current_value == test_code}")
                
                if current_value == test_code:
                    print(f"   ✅ {method['name']} 成功！")
                    code_success = True
                    break
                else:
                    print(f"   ❌ {method['name']} 失败")
                    
            except Exception as e:
                print(f"   ❌ {method['name']} 出错: {e}")
        
        if not code_success:
            print("\n❌ 所有信用代码输入方法都失败")
            print("请手动输入信用代码...")
            input("输入完成后按回车继续...")
        
        # 最终验证
        print("\n4. 最终验证...")
        final_company = (company_input.value or company_input.text or "").strip()
        final_code = (code_input.value or code_input.text or "").strip()
        
        print(f"公司名称: '{final_company}' (期望: '{test_company}')")
        print(f"信用代码: '{final_code}' (期望: '{test_code}')")
        
        company_ok = final_company == test_company
        code_ok = final_code == test_code
        
        print(f"公司名称匹配: {company_ok}")
        print(f"信用代码匹配: {code_ok}")
        
        if company_ok and code_ok:
            print("🎉 两个字段都输入成功！")
        elif company_ok:
            print("⚠️ 只有公司名称成功")
        elif code_ok:
            print("⚠️ 只有信用代码成功")
        else:
            print("❌ 两个字段都失败")
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        logger.exception("详细错误信息")
        
    finally:
        input("\n按回车键关闭浏览器...")
        scraper.close()

def clear_input_completely(scraper, input_element):
    """彻底清空输入框"""
    try:
        # 方法1: 直接clear
        input_element.clear()
    except:
        pass
    
    try:
        # 方法2: JavaScript清空
        scraper.page.run_js("arguments[0].value = '';", input_element)
    except:
        pass
    
    try:
        # 方法3: 全选删除
        input_element.click()
        time.sleep(0.1)
        scraper.page.key_down(['ctrl', 'a'])
        time.sleep(0.1)
        scraper.page.key_up(['ctrl', 'a'])
        time.sleep(0.1)
        scraper.page.key_down('Delete')
        time.sleep(0.1)
        scraper.page.key_up('Delete')
    except:
        pass

def simple_input(scraper, input_element, value):
    """方法1: 简单input"""
    input_element.input(value)

def clear_and_input(scraper, input_element, value):
    """方法2: clear + input"""
    input_element.clear()
    time.sleep(0.2)
    input_element.input(value)

def js_set_value(scraper, input_element, value):
    """方法3: JavaScript setValue"""
    scraper.page.run_js(f"arguments[0].value = '{value}';", input_element)

def js_with_events(scraper, input_element, value):
    """方法4: JavaScript + 事件"""
    js_code = f"""
    var input = arguments[0];
    var value = '{value}';
    
    input.focus();
    input.value = '';
    input.value = value;
    
    var events = ['focus', 'input', 'change', 'blur', 'keydown', 'keyup'];
    events.forEach(function(eventType) {{
        var event = new Event(eventType, {{
            bubbles: true,
            cancelable: true
        }});
        input.dispatchEvent(event);
    }});
    """
    scraper.page.run_js(js_code, input_element)

def keyboard_input(scraper, input_element, value):
    """方法5: 模拟键盘输入"""
    input_element.click()
    time.sleep(0.2)
    
    # 全选删除
    scraper.page.key_down(['ctrl', 'a'])
    time.sleep(0.1)
    scraper.page.key_up(['ctrl', 'a'])
    time.sleep(0.1)
    scraper.page.key_down('Delete')
    time.sleep(0.1)
    scraper.page.key_up('Delete')
    time.sleep(0.2)
    
    # 使用send_keys输入
    input_element.input(value)

def clipboard_paste(scraper, input_element, value):
    """方法6: 剪贴板粘贴"""
    import pyperclip
    pyperclip.copy(value)
    
    input_element.click()
    time.sleep(0.2)
    
    # 全选
    scraper.page.key_down(['ctrl', 'a'])
    time.sleep(0.1)
    scraper.page.key_up(['ctrl', 'a'])
    time.sleep(0.1)
    
    # 粘贴
    scraper.page.key_down(['ctrl', 'v'])
    time.sleep(0.1)
    scraper.page.key_up(['ctrl', 'v'])

def char_by_char_keyboard(scraper, input_element, value):
    """方法7: 逐字符键盘输入"""
    input_element.click()
    time.sleep(0.2)
    
    # 清空
    scraper.page.key_down(['ctrl', 'a'])
    time.sleep(0.1)
    scraper.page.key_up(['ctrl', 'a'])
    time.sleep(0.1)
    scraper.page.key_down('Delete')
    time.sleep(0.1)
    scraper.page.key_up('Delete')
    time.sleep(0.2)
    
    # 逐字符输入
    for char in value:
        scraper.page.key_down(char)
        time.sleep(0.05)
        scraper.page.key_up(char)
        time.sleep(0.05)

if __name__ == "__main__":
    test_all_input_methods() 