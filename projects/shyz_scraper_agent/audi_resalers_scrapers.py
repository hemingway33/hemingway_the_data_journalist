import time
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

def scrape_audi_dealers_with_selenium():
    """
    使用Selenium模拟浏览器操作，遍历全国城市，获取奥迪全部经销商信息。
    """
    # 使用webdriver-manager自动配置ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    # 目标网址
    url = "https://www.audi.cn/zh/new-dealerprocity.html"
    
    all_dealers_data = []
    processed_dealers = set() # 用于存储已处理的经销商名称，防止重复添加

    try:
        print("🚀 正在启动浏览器并访问奥迪官网...")
        driver.get(url)
        # 设置一个较长的等待时间，确保页面元素能加载出来
        wait = WebDriverWait(driver, 20)

        # 1. 处理可能出现的Cookie弹窗
        try:
            # 等待“接受全部”按钮出现并点击
            accept_button = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            accept_button.click()
            print("✅ 已接受Cookie策略。")
        except TimeoutException:
            print("🟡 未检测到Cookie弹窗，继续执行。")

        # 2. 打开省份/城市选择列表
        print("🔍 正在打开省份/城市列表...")
        # 注意：这里的选择器需要根据实际页面结构来确定
        province_dropdown_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".audi-dealer-search-entry-point__province-name-container")))
        province_dropdown_button.click()
        print("✅ 已打开省份/城市列表。")

        # 3. 获取所有省份的列表项
        # 等待省份列表加载完成
        provinces_list_container = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".audi-dealer-search-dropdown-list__container")))
        # 获取所有的省份元素
        province_elements = provinces_list_container.find_elements(By.TAG_NAME, "li")
        province_count = len(province_elements)
        print(f"共找到 {province_count} 个省份/直辖市。")
        
        # 遍历所有省份进行点击和数据提取
        for i in range(province_count):
            # 每次循环都重新获取省份列表，以避免“StaleElementReferenceException”
            provinces_list_container = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".audi-dealer-search-dropdown-list__container")))
            province_elements = provinces_list_container.find_elements(By.TAG_NAME, "li")
            
            # 如果当前省份元素不可见，需要滚动
            province_to_click = province_elements[i]
            driver.execute_script("arguments[0].scrollIntoView(true);", province_to_click)
            time.sleep(0.5) # 等待滚动动画

            province_name = province_to_click.text
            print(f"\n--- 正在处理省份: {province_name} ({i+1}/{province_count}) ---")

            # 点击省份
            province_to_click.click()
            # 等待经销商列表更新（给2秒的固定等待时间，等待JS加载数据）
            time.sleep(2) 

            # 4. 提取当前省份下的所有经销商信息
            dealer_list_items = driver.find_elements(By.CSS_SELECTOR, ".audi-dealer-search-result-list__item")
            print(f"在 {province_name} 找到 {len(dealer_list_items)} 家经销商。")

            for dealer_item in dealer_list_items:
                try:
                    name = dealer_item.find_element(By.CSS_SELECTOR, ".audi-dealer-card__name").text
                    # 使用经销商全称作为唯一标识，如果已经处理过则跳过
                    if name in processed_dealers:
                        continue
                    
                    address = dealer_item.find_element(By.CSS_SELECTOR, ".audi-dealer-card__address").text
                    # 电话号码可能不止一个，这里获取所有电话
                    phones = [phone.text for phone in dealer_item.find_elements(By.CSS_SELECTOR, ".audi-dealer-card-phone-number__number")]
                    
                    dealer_info = {
                        "省份": province_name.split("\n")[0], # 处理可能的多行文本
                        "城市": "", # 网站结构中城市信息和省份合并，暂留空
                        "经销商全称": name,
                        "地址": address,
                        "电话": " / ".join(phones)
                    }
                    all_dealers_data.append(dealer_info)
                    processed_dealers.add(name) # 添加到已处理集合
                except Exception as e:
                    print(f"  - 提取单个经销商信息时出错: {e}")
            
            # 操作完成后，需要再次点击打开省份列表，以便选择下一个
            if i < province_count - 1:
                wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".audi-dealer-search-entry-point__province-name-container"))).click()
                # 等待列表重新可见
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".audi-dealer-search-dropdown-list__container")))


        print(f"\n✔️ 数据抓取完成！共获得 {len(all_dealers_data)} 条不重复的经销商信息。")
        return all_dealers_data

    except Exception as e:
        print(f"❌ 程序执行过程中发生严重错误: {e}")
        return None
    finally:
        # 确保浏览器在程序结束时关闭
        print("👋 正在关闭浏览器...")
        driver.quit()

if __name__ == "__main__":
    dealers = scrape_audi_dealers_with_selenium()

    if dealers:
        # 保存为JSON文件
        json_path = 'audi_dealers_selenium.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dealers, f, ensure_ascii=False, indent=4)
        print(f"\n🎉 全部经销商数据已成功保存到文件: {json_path}")

        # 保存为Excel文件
        try:
            df = pd.DataFrame(dealers)
            excel_path = 'audi_dealers_selenium.xlsx'
            df.to_excel(excel_path, index=False)
            print(f"🎉 同时已将数据成功保存到Excel文件: {excel_path}")
        except ImportError:
            print("\n💡 提示: 如需保存为Excel文件，请确保已安装pandas库 (pip install pandas)。")