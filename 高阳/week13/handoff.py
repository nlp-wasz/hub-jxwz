    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    session = AdvancedSQLiteSession(
        session_id="test1",
        db_path="./assert/conversations.db",
        create_tables=True
    )

    stock_agent = Agent(
        name="Stock Agent",
        instructions=instructions,
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
    )
    result = Runner.run_streamed(triage_agent, input=content, session=session)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
