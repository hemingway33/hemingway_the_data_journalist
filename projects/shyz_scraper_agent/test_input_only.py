#!/usr/bin/env python3
"""
专门测试输入功能的脚本
验证是否能够自动输入公司名称和信用代码
"""

import time
import logging
from shyz_scraper_drission import SHYZScraperDrission, CompanyQuery

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_input_functionality():
    """专门测试输入功能"""
    print("=== 输入功能测试 ===")
    print("测试目标：验证能否自动输入公司名称和信用代码")
    print("公司名称: 上海遇圆实业有限公司")
    print("信用代码: 91310000MAE97Y0K5Y")
    print()
    
    # 创建爬虫实例 - 显示浏览器便于观察
    scraper = SHYZScraperDrission(
        headless=False,  # 显示浏览器便于观察
        timeout=30
    )
    
    try:
        print("1. 导航到查询页面...")
        success = scraper.navigate_to_query_page()
        
        if success:
            print("✅ 成功进入查询页面")
            
            # 创建查询对象
            query = CompanyQuery(
                company_name="上海遇圆实业有限公司",
                credit_code="91310000MAE97Y0K5Y"
            )
            
            print("\n2. 测试公司名称输入...")
            
            # 找到公司名称输入框
            company_input_selectors = [
                "xpath://input[contains(@placeholder, '单位名称')]",
                "xpath://label[contains(text(), '单位名称')]/..//input",
                "xpath://label[contains(text(), '单位名称')]/following-sibling::input"
            ]
            
            company_input = None
            for selector in company_input_selectors:
                try:
                    company_input = scraper.page.ele(selector)
                    if company_input:
                        logger.info(f"找到单位名称输入框: {selector}")
                        break
                except Exception:
                    continue
            
            if not company_input:
                print("❌ 无法找到单位名称输入框")
                return
            
            # 测试多种输入方法
            print("尝试输入公司名称...")
            
            # 显示输入前的状态
            print(f"输入前状态: '{company_input.value}'")
            
            # 方法1: JavaScript完整事件
            try:
                print("   尝试方法1: JavaScript完整事件...")
                js_code = f"""
                var input = arguments[0];
                var value = '{query.company_name}';
                
                // 聚焦元素
                input.focus();
                
                // 清空现有值
                input.value = '';
                
                // 设置新值
                input.value = value;
                
                // 触发完整的事件链
                var events = ['focus', 'input', 'change', 'blur'];
                events.forEach(function(eventType) {{
                    var event = new Event(eventType, {{
                        bubbles: true,
                        cancelable: true
                    }});
                    input.dispatchEvent(event);
                }});
                
                // 返回当前值用于验证
                return input.value;
                """
                result = scraper.page.run_js(js_code, company_input)
                time.sleep(2)
                current_value = company_input.value
                print(f"   方法1结果: '{current_value}'")
                
                if current_value == query.company_name:
                    print("   ✅ 方法1成功！")
                    company_input_success = True
                else:
                    print("   ❌ 方法1失败")
                    company_input_success = False
                    
            except Exception as e:
                print(f"   ❌ 方法1出错: {e}")
                company_input_success = False
            
            # 如果方法1失败，尝试方法2
            if not company_input_success:
                try:
                    print("   尝试方法2: 模拟键盘输入...")
                    # 点击输入框
                    company_input.click()
                    time.sleep(0.5)
                    
                    # 全选并删除
                    scraper.page.key_down(['ctrl', 'a'])
                    time.sleep(0.1)
                    scraper.page.key_up(['ctrl', 'a'])
                    time.sleep(0.1)
                    scraper.page.key_down('Delete')
                    time.sleep(0.1)
                    scraper.page.key_up('Delete')
                    time.sleep(0.5)
                    
                    # 逐字符输入
                    for char in query.company_name:
                        scraper.page.key_down(char)
                        time.sleep(0.05)
                        scraper.page.key_up(char)
                        time.sleep(0.05)
                    
                    time.sleep(2)
                    current_value = company_input.value
                    print(f"   方法2结果: '{current_value}'")
                    
                    if current_value == query.company_name:
                        print("   ✅ 方法2成功！")
                        company_input_success = True
                    else:
                        print("   ❌ 方法2失败")
                        
                except Exception as e:
                    print(f"   ❌ 方法2出错: {e}")
            
            # 如果还是失败，尝试剪贴板方法
            if not company_input_success:
                try:
                    print("   尝试方法3: 剪贴板粘贴...")
                    import pyperclip
                    pyperclip.copy(query.company_name)
                    
                    company_input.click()
                    time.sleep(0.5)
                    
                    # 全选并粘贴
                    scraper.page.key_down(['ctrl', 'a'])
                    time.sleep(0.1)
                    scraper.page.key_up(['ctrl', 'a'])
                    time.sleep(0.1)
                    scraper.page.key_down(['ctrl', 'v'])
                    time.sleep(0.1)
                    scraper.page.key_up(['ctrl', 'v'])
                    time.sleep(2)
                    
                    current_value = company_input.value
                    print(f"   方法3结果: '{current_value}'")
                    
                    if current_value == query.company_name:
                        print("   ✅ 方法3成功！")
                        company_input_success = True
                    else:
                        print("   ❌ 方法3失败")
                        
                except Exception as e:
                    print(f"   ❌ 方法3出错: {e}")
            
            if company_input_success:
                print("✅ 公司名称输入成功")
            else:
                print("❌ 公司名称输入失败")
                print("请手动输入公司名称以继续测试...")
                input("输入完成后按回车键继续...")
            
            print("\n3. 测试信用代码输入...")
            
            # 找到信用代码输入框
            code_input_selectors = [
                "xpath://input[contains(@placeholder, '社会信用代码')]",
                "xpath://input[contains(@placeholder, '信用代码')]", 
                "xpath://label[contains(text(), '社会信用代码')]/..//input",
                "xpath://label[contains(text(), '社会信用代码')]/following-sibling::input"
            ]
            
            code_input = None
            for selector in code_input_selectors:
                try:
                    code_input = scraper.page.ele(selector)
                    if code_input:
                        logger.info(f"找到社会信用代码输入框: {selector}")
                        break
                except Exception:
                    continue
            
            if not code_input:
                print("❌ 无法找到社会信用代码输入框")
                return
            
            # 测试信用代码输入（使用相同的方法）
            print("尝试输入信用代码...")
            print(f"输入前状态: '{code_input.value}'")
            
            # 方法1: JavaScript完整事件
            try:
                print("   尝试方法1: JavaScript完整事件...")
                js_code = f"""
                var input = arguments[0];
                var value = '{query.credit_code}';
                
                input.focus();
                input.value = '';
                input.value = value;
                
                var events = ['focus', 'input', 'change', 'blur'];
                events.forEach(function(eventType) {{
                    var event = new Event(eventType, {{
                        bubbles: true,
                        cancelable: true
                    }});
                    input.dispatchEvent(event);
                }});
                
                return input.value;
                """
                result = scraper.page.run_js(js_code, code_input)
                time.sleep(2)
                current_code = code_input.value
                print(f"   方法1结果: '{current_code}'")
                
                if current_code == query.credit_code:
                    print("   ✅ 方法1成功！")
                    code_input_success = True
                else:
                    print("   ❌ 方法1失败")
                    code_input_success = False
                    
            except Exception as e:
                print(f"   ❌ 方法1出错: {e}")
                code_input_success = False
            
            if code_input_success:
                print("✅ 信用代码输入成功")
            else:
                print("❌ 信用代码输入失败")
                print("请手动输入信用代码...")
                input("输入完成后按回车键继续...")
            
            print("\n4. 最终验证...")
            final_company = company_input.value
            final_code = code_input.value
            print(f"公司名称: '{final_company}'")
            print(f"信用代码: '{final_code}'")
            
            if final_company == query.company_name and final_code == query.credit_code:
                print("🎉 输入功能测试完全成功！")
            elif final_company == query.company_name:
                print("⚠️ 公司名称成功，信用代码需要手动输入")
            elif final_code == query.credit_code:
                print("⚠️ 信用代码成功，公司名称需要手动输入")
            else:
                print("❌ 两个字段都需要手动输入")
            
        else:
            print("❌ 无法进入查询页面")
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        logger.exception("详细错误信息")
        
    finally:
        input("\n按回车键关闭浏览器...")
        scraper.close()
        print("测试完成!")

if __name__ == "__main__":
    test_input_functionality() 