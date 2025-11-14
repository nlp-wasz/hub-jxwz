## 1. 数据管理接口
### 1.1 知识库管理
#### POST /api/v1/knowledge-bases
    Request:
    {
        "name": "string",                    # 知识库名称
        "description": "string",             # 描述
        "permissions": {                     # 权限设置
            "read_users": ["user1", "user2"],
            "write_users": ["user1", "admin"],
            "public_read": false
        },
        "tags": ["金融", "乳制品", "公司研究"]
    }
    
    Response:
    {
        "code": 200,
        "message": "success",
        "data": {
            "knowledge_base_id": "kb_ilid_001",
            "name": "伊利股份研究报告库",
            "created_time": "2024-01-01T00:00:00Z",
            "document_count": 0
        }
    }

### 1.2 文档上传接口
#### POST /api/v1/documents/upload
    Request (multipart/form-data):
    - knowledge_base_id: string (required)
      - file: file (required, pdf/docx/txt/png/jpg)
      - document_name: string (optional)
      - tags: list[string] (optional)
      - chunk_strategy: "fixed_size" | "semantic" (default: "semantic")
      - chunk_size: int (default: 1000)
      - chunk_overlap: int (default: 200)
    
    Response:
    {
        "code": 200,
        "message": "success",
        "data": {
            "document_id": "doc_ilid_20200427_001",
            "document_name": "公司研究伊利股份-护城河稳固平台化发展未来可期-20200427",
            "knowledge_base_id": "kb_ilid_001",
            "status": "pending",  # pending, processing, completed, failed
            "file_size": 1024000,
            "pages": 22,
            "upload_time": "2024-01-01T00:00:00Z",
            "parse_task_id": "parse_task_001"
        }
    }

### 1.3 文档状态查询
#### GET /api/v1/documents/{document_id}/status
    Response:
    {
        "code": 200,
        "message": "success",
        "data": {
            "document_id": "doc_ilid_20200427_001",
            "status": "processing",  # pending, processing, completed, failed
            "progress": 65,          # 处理进度百分比
            "pages_processed": 15,
            "total_pages": 22,
            "text_chunks_created": 45,
            "images_extracted": 8,
            "current_step": "ocr_processing",  # file_uploaded, ocr_processing, chunking, embedding, completed
            "estimated_remaining_time": 120,   # 预估剩余时间(秒)
            "error_message": null
        }
    }

### 1.4 文档列表查询
#### GET /api/v1/documents?knowledge_base_id={kb_id}&page=1&size=20&status=completed
    Response:
    {
        "code": 200,
        "message": "success",
        "data": {
            "documents": [
                {
                    "document_id": "doc_ilid_20200427_001",
                    "document_name": "公司研究伊利股份-护城河稳固平台化发展未来可期-20200427",
                    "knowledge_base_id": "kb_ilid_001",
                    "status": "completed",
                    "file_size": 1024000,
                    "pages": 22,
                    "text_chunks": 68,
                    "images": 15,
                    "upload_time": "2024-01-01T00:00:00Z",
                    "process_complete_time": "2024-01-01T01:30:00Z",
                    "tags": ["公司研究", "乳制品", "开源证券"]
                }
            ],
            "total": 1,
            "page": 1,
            "size": 20
        }
    }

### 1.5 文档删除
#### DELETE /api/v1/documents/{document_id}
    Response:
    {
        "code": 200,
        "message": "success",
        "data": {
            "document_id": "doc_ilid_20200427_001",
            "status": "processing",  # pending, processing, completed, failed
            "progress": 65,          # 处理进度百分比
            "pages_processed": 15,
            "total_pages": 22,
            "text_chunks_created": 45,
            "images_extracted": 8,
            "current_step": "ocr_processing",
            "estimated_remaining_time": 120,   # 预估剩余时间(秒)
            "error_message": null
        }
    }

## 2. 多模态检索接口
### 2.1 多模态混合检索
#### POST /api/v1/retrieve/multimodal
    Request:
    {
        "knowledge_base_id": "kb_ilid_001",
        "query": "伊利股份在常温奶市场的占有率是多少？与蒙牛相比如何？",
        "query_image": "base64_string",     # 可选的查询图片
        "modalities": ["text", "image"],    # 检索模态
        "top_k": 10,
        "score_threshold": 0.7,
        "filters": {
            "document_ids": ["doc_ilid_20200427_001"],
            "tags": ["市场占有率", "竞争格局"],
            "date_range": {
                "start": "2020-01-01",
                "end": "2020-12-31"
            },
            "page_range": {
                "start": 1,
                "end": 10
            }
        },
        "rerank": true,  # 是否重排序
        "diversify": true # 是否多样化结果
    }
    
    Response:
    {
        "code": 200,
        "message": "success",
        "data": {
            "text_results": [
                {
                    "content": "2019年中国常温奶CR2约65%，伊利股份和蒙牛乳业呈现双寡头格局。",
                    "document_id": "doc_ilid_20200427_001",
                    "document_name": "公司研究伊利股份-护城河稳固平台化发展未来可期-20200427",
                    "page_number": 8,
                    "chunk_id": "chunk_001",
                    "score": 0.89,
                    "metadata": {
                        "section_title": "行业呈双寡头格局，行业集中度持续提升",
                        "font_size": 10,
                        "position": {"x": 50, "y": 200, "width": 400, "height": 30},
                        "chunk_type": "paragraph",
                        "importance_score": 0.8
                    },
                    "surrounding_context": {
                        "previous_chunk": "乳制品行业集中度持续提升...",
                        "next_chunk": "全国性乳制品企业处于常态化竞争状态..."
                    }
                }
            ],
            "image_results": [
                {
                    "image_id": "img_ilid_001",
                    "document_id": "doc_ilid_20200427_001",
                    "document_name": "公司研究伊利股份-护城河稳固平台化发展未来可期-20200427",
                    "page_number": 8,
                    "image_caption": "图11：2019年中国常温奶CR2约65%，显示伊利和蒙牛的市场占有率对比",
                    "image_path": "/storage/images/img_ilid_001.png",
                    "image_base64": "base64_string",
                    "score": 0.78,
                    "bounding_box": {"x": 100, "y": 300, "width": 300, "height": 200},
                    "text_context": "围绕图片的文本内容：乳制品行业集中度持续提升，呈双寡头格局...",
                    "ocr_text": "2019年中国常温奶CR2约65%\n伊利 蒙牛\n40.00%\n35.00%\n30.00%",
                    "metadata": {
                        "figure_type": "bar_chart",
                        "data_categories": ["市场占有率"],
                        "entities": ["伊利", "蒙牛", "常温奶"]
                    }
                }
            ],
            "query_analysis": {
                "intent": "market_share_comparison",
                "entities": ["伊利股份", "蒙牛", "常温奶", "市场占有率"],
                "modality_weights": {"text": 0.7, "image": 0.3}
            },
            "retrieval_metrics": {
                "total_time": 0.45,
                "text_retrieval_time": 0.25,
                "image_retrieval_time": 0.20,
                "rerank_time": 0.10,
                "text_candidates": 150,
                "image_candidates": 25
            }
        }
    }

### 2.2 图表数据检索
#### POST /api/v1/retrieve/chart-data
    Request:
    {
        "knowledge_base_id": "kb_ilid_001",
        "chart_query": "获取伊利股份2017-2021年的营业收入和净利润数据",
        "chart_types": ["table", "line_chart", "bar_chart"],
        "filters": {
            "document_ids": ["doc_ilid_20200427_001"],
            "data_categories": ["财务数据", "营收", "利润"]
        }
    }
    
    Response:
    {
        "code": 200,
        "message": "success",
        "data": {
            "structured_data": [
                {
                    "year": 2017,
                    "revenue": 67547,
                    "net_profit": 6001,
                    "revenue_growth": "12.0%",
                    "profit_growth": "6.0%"
                },
                {
                    "year": 2018,
                    "revenue": 78976,
                    "net_profit": 6440,
                    "revenue_growth": "16.9%",
                    "profit_growth": "7.3%"
                }
            ],
            "source_images": [
                {
                    "image_id": "img_ilid_financial_001",
                    "page_number": 20,
                    "caption": "财务预测摘要表格",
                    "confidence": 0.92
                }
            ],
            "data_sources": [
                {
                    "type": "extracted_table",
                    "location": "page_20_table_1",
                    "extraction_method": "ocr_table_detection"
                }
            ]
        }
    }

## 3. 多模态问答接口
### 3.1 多模态问答
#### POST /api/v1/chat/multimodal
    Request:
    {
        "knowledge_base_id": "kb_ilid_001",
        "query": "根据报告中的图表，分析伊利股份在常温奶市场的竞争地位和发展趋势",
        "query_image": "base64_string",  # 可选的查询图片
        "chat_history": [
            {
                "role": "user",
                "content": "伊利股份的主要业务是什么？",
                "timestamp": "2024-01-01T10:00:00Z"
            },
            {
                "role": "assistant",
                "content": "伊利股份的主要业务是液态奶，2019H1占比达80.2%，其次是奶粉和冷饮产品。",
                "timestamp": "2024-01-01T10:01:00Z",
                "sources": [
                    {
                        "type": "text",
                        "document_id": "doc_ilid_20200427_001",
                        "page_number": 5,
                        "content": "液态乳是伊利股份的主要收入来源，2019H1占比达80.2%"
                    }
                ]
            }
        ],
        "retrieval_config": {
            "top_k": 15,
            "score_threshold": 0.6,
            "modalities": ["text", "image", "table"],
            "enable_rerank": true,
            "diversify_results": true
        },
        "generation_config": {
            "model": "qwen-vl-max",  # qwen-vl-plus, qwen-vl-max
            "temperature": 0.7,
            "max_tokens": 2000,
            "include_sources": true,
            "include_reasoning": true,
            "response_format": "structured"  # structured, free_form
        },
        "stream": false
    }
    
    Response:
    {
        "code": 200,
        "message": "success",
        "data": {
            "answer": "根据研究报告分析，伊利股份在常温奶市场具有稳固的竞争地位：\n\n1. **市场格局**：2019年中国常温奶市场呈现典型的双寡头格局，CR2（伊利+蒙牛）合计达到约65%。从图表11可以看出，伊利和蒙牛共同主导市场。\n\n2. **竞争优势**：伊利股份通过渠道扁平化、产品线丰富和品牌力强建立了稳固的竞争壁垒。公司在2019H1常温液态乳市场渗透率达83.9%，在三四线城市渗透率达86.2%。\n\n3. **发展趋势**：尽管行业竞争激烈，但伊利通过不断的产品创新（如安慕希系列新品）和渠道深耕，持续扩大市场份额。公司2015年以来营收逐渐拉大与蒙牛的差距。\n\n4. **未来展望**：疫情后行业集中度可能加速提升，伊利凭借强大的渠道力和品牌力有望进一步扩大市场份额。",
            "sources": {
                "text_sources": [
                    {
                        "content": "2019年中国常温奶CR2约65%，伊利股份和蒙牛乳业呈现双寡头格局...",
                        "document_id": "doc_ilid_20200427_001",
                        "document_name": "公司研究伊利股份-护城河稳固平台化发展未来可期-20200427",
                        "page_number": 8,
                        "score": 0.89,
                        "chunk_id": "chunk_001"
                    },
                    {
                        "content": "伊利股份具有较强的渠道力，可依靠渠道优势迅速将产品进行全国扩张...",
                        "document_id": "doc_ilid_20200427_001",
                        "page_number": 10,
                        "score": 0.85
                    }
                ],
                "image_sources": [
                    {
                        "image_id": "img_ilid_001",
                        "document_id": "doc_ilid_20200427_001",
                        "page_number": 8,
                        "image_caption": "图11：2019年中国常温奶CR2约65%",
                        "image_path": "/storage/images/img_ilid_001.png",
                        "score": 0.92,
                        "analysis": "该条形图清晰显示了伊利和蒙牛在常温奶市场的占有率对比"
                    },
                    {
                        "image_id": "img_ilid_002", 
                        "document_id": "doc_ilid_20200427_001",
                        "page_number": 10,
                        "image_caption": "图14：2015年以来伊利股份营收拉大与蒙牛乳业的差距",
                        "score": 0.78
                    }
                ]
            },
            "reasoning_chain": [
                {
                    "step": 1,
                    "action": "query_understanding",
                    "content": "识别用户需要分析伊利股份在常温奶市场的竞争地位和发展趋势，重点关注图表数据",
                    "confidence": 0.95
                },
                {
                    "step": 2, 
                    "action": "multimodal_retrieval",
                    "content": "检索到关于市场格局、竞争地位、发展趋势的相关文本和图表",
                    "confidence": 0.88
                },
                {
                    "step": 3,
                    "action": "chart_analysis",
                    "content": "分析图11的市场占有率数据和图14的营收对比趋势",
                    "confidence": 0.90
                },
                {
                    "step": 4,
                    "action": "synthesis",
                    "content": "综合文本描述和图表数据，构建完整的竞争分析框架",
                    "confidence": 0.92
                }
            ],
            "structured_insights": {
                "market_position": "双寡头领导者",
                "market_share": "约35%",
                "competitive_advantages": ["渠道力", "产品创新", "品牌力"],
                "growth_trend": "稳步提升",
                "key_metrics": {
                    "market_penetration": "83.9%",
                    "low_tier_penetration": "86.2%"
                }
            },
            "metrics": {
                "retrieval_time": 0.45,
                "generation_time": 2.34,
                "total_time": 2.79,
                "tokens_used": 1567,
                "sources_used": {"text": 3, "images": 2}
            }
        }
    }

## 4. 错误处理
### 4.1 统一错误响应 
    {
        "code": 400,
        "message": "Invalid request parameters",
        "data": null,
        "details": {
            "error_type": "VALIDATION_ERROR",
            "field_errors": {
                "knowledge_base_id": "Knowledge base not found",
                "query": "Query cannot be empty"
            },
            "suggestions": [
                "Check if the knowledge base exists",
                "Provide a valid query string"
            ]
        }
    }