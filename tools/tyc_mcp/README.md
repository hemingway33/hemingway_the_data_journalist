# Tianyancha MCP Server

This is an MCP (Model Context Protocol) server for interacting with Tianyancha (天眼查) data.

## Features

- Company search by keyword
- Detailed company information retrieval
- Company list resource access

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure authentication:
   - You need to provide your Tianyancha authentication token
   - Update the `auth_token` in the `main()` function of `server.py`

## Usage

Run the server:
```bash
python server.py
```

## Available Tools

1. `search_company(keyword: str) -> List[CompanyInfo]`
   - Search for companies by keyword
   - Returns a list of matching companies with basic information

2. `get_company_details(company_id: str) -> CompanyInfo`
   - Get detailed information about a specific company
   - Returns comprehensive company information

## Available Resources

1. `company_list`
   - Access a list of companies from recent searches

## Note

This is a basic implementation. You may need to:
- Implement proper HTML parsing for the responses
- Add error handling for rate limiting
- Add caching mechanisms
- Implement additional tools and resources as needed 