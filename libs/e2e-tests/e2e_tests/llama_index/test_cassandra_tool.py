import uuid

import cassio
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.cassandra.base import CassandraDatabaseToolSpec
from llama_index.tools.cassandra.cassandra_database_wrapper import (
    CassandraDatabase,
)


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

    spec = CassandraDatabaseToolSpec(db=db)

    tools = spec.to_tool_list()
    for tool in tools:
        print(tool.metadata.name)

    llm = OpenAI(model="gpt-4o")
    agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)

    response = agent.chat(
        "What is the user_id of the user named 'my_user' in table default_keyspace.tool_table_users?"
    )
    print(response)
    assert response is not None
    assert str(user_id) in str(response)
