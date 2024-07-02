import uuid

import cassio
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.cassandra_database.tool import (
    GetSchemaCassandraDatabaseTool,
    GetTableDataCassandraDatabaseTool,
    QueryCassandraDatabaseTool,
)
from langchain_community.utilities.cassandra_database import CassandraDatabase
from langchain_openai import ChatOpenAI


def test_tool_with_openai_tool(cassandra):
    session = cassio.config.resolve_session()
    session.execute("DROP TABLE IF EXISTS default_keyspace.tool_table_users;")

    session.execute(
        """
        CREATE TABLE IF NOT EXISTS default_keyspace.tool_table_users (
        user_id UUID PRIMARY KEY ,
        user_name TEXT ,
        password TEXT
    );
    """
    )
    session.execute(
        """
    CREATE INDEX user_name
   ON default_keyspace.tool_table_users (user_name);
    """
    )

    user_id = uuid.uuid4()
    session.execute(
        f"""
        INSERT INTO default_keyspace.tool_table_users (user_id, user_name)
        VALUES ({user_id}, 'my_user');
    """
    )
    db = CassandraDatabase()

    query_tool = QueryCassandraDatabaseTool(db=db)
    schema_tool = GetSchemaCassandraDatabaseTool(db=db)
    select_data_tool = GetTableDataCassandraDatabaseTool(db=db)

    tools = [schema_tool, select_data_tool, query_tool]

    model = ChatOpenAI(model="gpt-4o")

    prompt = hub.pull("hwchase17/openai-tools-agent")

    agent = create_openai_tools_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke(
        {
            "input": "What is the user_id of the user named 'my_user' "
            "in table default_keyspace.tool_table_users?"
        }
    )
    print(response)
    assert response is not None
    assert str(user_id) in str(response)
