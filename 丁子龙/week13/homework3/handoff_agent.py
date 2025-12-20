async def chat(user_name: str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
    # 对话管理，通过session id
    if session_id:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id, task)

    # 对话记录，存关系型数据库
    append_message2db(session_id, "user", content)

    # 获取system message，需要传给大模型，并不能给用户展示
    instructions = get_init_message(task)

    # agent 初始化
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    # mcp tools 选择
    if not tools or len(tools) == 0:
        tool_mcp_tools_filter: Optional[ToolFilterStatic] = None
    else:
        tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=tools)
    mcp_server = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,
        client_session_timeout_seconds=20,
    )

    # openai-agent支持的session存储，存储对话的历史状态
    session = AdvancedSQLiteSession(
        session_id=session_id,  # 与 系统中的对话id 关联，存储在关系型数据库中
        db_path="./assert/conversations.db",
        create_tables=True
    )
    stock_agent = Agent(
        name="Stock Agent",
        instructions=instructions,
        mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=external_client,
        ),
        # tool_use_behavior="stop_on_first_tool",
        model_settings=ModelSettings(parallel_tool_calls=False)
    )

    chat_agent = Agent(
        name="Chat Agent",
        instructions=instructions,
        # mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=external_client,
        ),
        # tool_use_behavior="stop_on_first_tool",
        model_settings=ModelSettings(parallel_tool_calls=False)
    )

    triage_agent = Agent(
        name="Triage Agent",
        model="qwen-max",
        instructions="您的任务是根据用户的问题查看是闲聊还是股票相关，判断应该将请求分派给 'Chat Agent' 还是 'Stock Agent'。",
        handoffs=[chat_agent, stock_agent],
        # input_guardrails=[
        #     InputGuardrail(guardrail_function=homework_guardrail),
        # ],
    )
    result = Runner.run_streamed(triage_agent, input=content, session=session)
    assistant_message = ""
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):  # 如果式大模型的回答
                if event.data.delta:
                    yield f"{event.data.delta}"  # sse 不断发给前端
                    assistant_message += event.data.delta

    # 这一条大模型回答，存储对话
    append_message2db(session_id, "assistant", assistant_message)