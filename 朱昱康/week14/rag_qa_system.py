#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG问答系统
整合RAG工具筛选和MCP工具执行，完成整个问答流程
"""

import json
import sys
import os
from typing import Dict, Any, List, Optional

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_tool_selector import ToolSelector
from mcp_service import MCPService

class RAGQASystem:
    """RAG问答系统，整合工具选择和执行"""
    
    def __init__(self):
        """初始化RAG问答系统"""
        self.tool_selector = ToolSelector()
        self.mcp_service = MCPService()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询，完成整个RAG流程
        
        Args:
            query (str): 用户查询
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        # 步骤1: 使用RAG选择最相关的工具
        selected_tools = self.tool_selector.select_tool(query, top_k=3)
        
        if not selected_tools:
            return {
                "success": False,
                "error": "未找到相关工具",
                "query": query
            }
        
        # 选择最相关的工具
        best_tool = selected_tools[0]
        tool_id = best_tool["tool_id"]
        
        # 步骤2: 尝试从查询中提取参数
        extracted_params = self.tool_selector.extract_parameters(query, tool_id)
        
        # 步骤3: 检查是否所有必需参数都已提取
        tool_schema = self.mcp_service.get_tool_schema(tool_id)
        required_params = tool_schema.get("required", []) if tool_schema else []
        missing_params = [param for param in required_params if param not in extracted_params]
        
        # 如果有缺失参数，请求用户提供
        if missing_params:
            return {
                "success": False,
                "error": f"缺少必需参数: {', '.join(missing_params)}",
                "tool_id": tool_id,
                "tool_name": best_tool["name"],
                "tool_description": best_tool["description"],
                "required_params": required_params,
                "extracted_params": extracted_params,
                "missing_params": missing_params,
                "query": query
            }
        
        # 步骤4: 执行工具函数
        execution_result = self.mcp_service.execute_tool(tool_id, extracted_params)
        
        # 步骤5: 汇总结果
        if execution_result["success"]:
            return {
                "success": True,
                "query": query,
                "selected_tool": {
                    "id": tool_id,
                    "name": best_tool["name"],
                    "description": best_tool["description"],
                    "similarity": best_tool["similarity"]
                },
                "parameters": extracted_params,
                "result": execution_result["result"],
                "all_candidates": selected_tools
            }
        else:
            return {
                "success": False,
                "error": execution_result.get("error", "未知错误"),
                "query": query,
                "selected_tool": {
                    "id": tool_id,
                    "name": best_tool["name"],
                    "description": best_tool["description"]
                },
                "parameters": extracted_params
            }
    
    def interactive_query(self, query: str, additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        交互式查询处理，允许用户提供额外参数
        
        Args:
            query (str): 用户查询
            additional_params (Optional[Dict[str, Any]]: 额外参数
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        # 使用RAG选择最相关的工具
        selected_tools = self.tool_selector.select_tool(query, top_k=1)
        
        if not selected_tools:
            return {
                "success": False,
                "error": "未找到相关工具",
                "query": query
            }
        
        # 选择最相关的工具
        best_tool = selected_tools[0]
        tool_id = best_tool["tool_id"]
        
        # 从查询中提取参数
        extracted_params = self.tool_selector.extract_parameters(query, tool_id)
        
        # 合并额外参数
        if additional_params:
            extracted_params.update(additional_params)
        
        # 检查是否所有必需参数都已提供
        tool_schema = self.mcp_service.get_tool_schema(tool_id)
        required_params = tool_schema.get("required", []) if tool_schema else []
        missing_params = [param for param in required_params if param not in extracted_params]
        
        if missing_params:
            return {
                "success": False,
                "error": f"缺少必需参数: {', '.join(missing_params)}",
                "tool_id": tool_id,
                "tool_name": best_tool["name"],
                "tool_description": best_tool["description"],
                "required_params": required_params,
                "provided_params": extracted_params,
                "missing_params": missing_params,
                "query": query
            }
        
        # 执行工具函数
        execution_result = self.mcp_service.execute_tool(tool_id, extracted_params)
        
        # 汇总结果
        if execution_result["success"]:
            return {
                "success": True,
                "query": query,
                "selected_tool": {
                    "id": tool_id,
                    "name": best_tool["name"],
                    "description": best_tool["description"],
                    "similarity": best_tool["similarity"]
                },
                "parameters": extracted_params,
                "result": execution_result["result"]
            }
        else:
            return {
                "success": False,
                "error": execution_result.get("error", "未知错误"),
                "query": query,
                "selected_tool": {
                    "id": tool_id,
                    "name": best_tool["name"],
                    "description": best_tool["description"]
                },
                "parameters": extracted_params
            }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        列出所有可用工具
        
        Returns:
            List[Dict[str, Any]]: 工具列表
        """
        return self.mcp_service.get_tool_list()


def main():
    """主函数，提供命令行交互界面"""
    qa_system = RAGQASystem()
    
    print("=== RAG公式解析与智能问答系统 ===")
    print("输入'help'查看可用工具列表")
    print("输入'quit'退出系统")
    print()
    
    while True:
        query = input("请输入您的问题: ").strip()
        
        if query.lower() == 'quit':
            print("感谢使用，再见！")
            break
        
        if query.lower() == 'help':
            tools = qa_system.list_tools()
            print("\n=== 可用工具列表 ===")
            for tool in tools:
                print(f"ID: {tool['id']}")
                print(f"名称: {tool['name']}")
                print(f"描述: {tool['description']}")
                print("-" * 50)
            continue
        
        # 处理查询
        result = qa_system.process_query(query)
        
        # 显示结果
        print("\n=== 查询结果 ===")
        print(f"查询: {result.get('query', 'N/A')}")
        
        if result.get("success"):
            selected_tool = result.get("selected_tool", {})
            print(f"选择工具: {selected_tool.get('name', 'N/A')} (相似度: {selected_tool.get('similarity', 0):.2f})")
            print(f"使用参数: {result.get('parameters', {})}")
            print(f"计算结果: {result.get('result', 'N/A')}")
        else:
            print(f"错误: {result.get('error', '未知错误')}")
            
            # 如果是缺少参数错误，显示所需参数
            if "missing_params" in result:
                print(f"\n工具 '{result.get('tool_name', 'N/A')}' 需要以下参数:")
                for param in result.get("required_params", []):
                    status = "✓" if param in result.get("extracted_params", {}) else "✗"
                    print(f"  {status} {param}")
                
                print("\n您可以尝试重新输入问题，包含所有必需参数，或使用交互模式:")
                print(f"interactive_query('{query}', {{参数名: 参数值, ...}})")


if __name__ == "__main__":
    main()