# 实物印章备案查询 Mock Scraper

Mock implementation of a Chinese Physical Seal Filing Inquiry system scraper.

## 功能特性 (Features)

- **连接管理**: 模拟与政府服务系统的连接
- **用户认证**: 支持用户登录/登出
- **多种查询方式**:
  - 按企业名称查询
  - 按印章编号查询  
  - 按企业代码查询
- **详细信息获取**: 获取印章完整详细信息
- **数据导出**: 支持查询结果导出
- **真实模拟**: 包含网络延迟、会话管理等真实场景

## 使用方法 (Usage)

### 基本使用

```python
from mock_scraper import SealInquiryScraper

# 创建爬虫实例
scraper = SealInquiryScraper()

# 连接系统
scraper.connect()

# 登录
scraper.login("admin", "password123")

# 查询企业
result = scraper.search_by_company_name("科技")

# 获取详细信息
if result["results"]:
    seal_id = result["results"][0]["seal_id"]
    details = scraper.get_seal_details(seal_id)

# 退出登录
scraper.logout()
```

### 运行演示

```bash
python mock_scraper.py
```

## API 接口

### SealInquiryScraper 类

#### connect() -> Dict
连接到印章备案查询系统

#### login(username: str, password: str) -> Dict
用户登录验证

#### search_by_company_name(company_name: str) -> Dict
按企业名称模糊查询

#### search_by_seal_id(seal_id: str) -> Dict  
按印章编号精确查询

#### search_by_company_code(company_code: str) -> Dict
按企业代码查询

#### get_seal_details(seal_id: str) -> Dict
获取印章详细信息

#### export_results(query_params: Dict) -> Dict
导出查询结果

#### logout() -> Dict
退出登录

## 数据结构

### SealRecord (印章记录)
- `seal_id`: 印章编号
- `company_name`: 企业名称  
- `company_code`: 企业代码
- `seal_type`: 印章类型 (公章、财务专用章、合同专用章、人事专用章、法人代表章)
- `filing_date`: 备案日期
- `status`: 状态 (正常、注销、变更、冻结)
- `authority`: 备案机关
- `contact_person`: 联系人
- `contact_phone`: 联系电话
- `remarks`: 备注

## 响应格式

所有 API 调用返回统一格式:

```json
{
  "status": "success|error",
  "message": "描述信息",
  "data": {...}
}
```

## 技术实现

- **纯 Python 标准库**: 无外部依赖
- **模拟真实场景**: 网络延迟、会话管理、数据分页
- **中文支持**: 完整的中文企业和印章数据
- **类型提示**: 完整的 Python 类型注解

## 注意事项

1. 这是一个 **模拟实现**，不连接真实的政府系统
2. 数据为随机生成的测试数据
3. 适用于开发测试和系统演示
4. 登录验证: 用户名 ≥ 4 位，密码 ≥ 6 位即可通过 