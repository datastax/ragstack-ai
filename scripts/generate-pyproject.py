import toml
import sys

from toml.decoder import InlineTableDict

def deep_update(base, updater):
    for key, value in updater.items():
        if isinstance(value, InlineTableDict):
            base[key] = value
        elif isinstance(value, dict):
            base[key] = deep_update(base.get(key, {}), value)
        elif value == "":
            del base[key]
        else:
            base[key] = value
    return base

def generate_pyproject(rag_stack_path, e2e_tests_path, update_file_path, output_file_path):
    with open(rag_stack_path, 'r') as rag_stack, open(e2e_tests_path, 'r') as e2e_tests, open(update_file_path, 'r') as update_file:
        rag_stack_data = toml.load(rag_stack)
        e2e_tests_data = toml.load(e2e_tests)
        update_data = toml.load(update_file)

        # grab all the RAGstack module dependencies from the root project.toml
        root_dependencies = rag_stack_data["tool"]["poetry"]["dependencies"]

        # remove the root `python` dependency
        if "python" in root_dependencies:
            del root_dependencies["python"]

        # remove `optional` flags if they exist
        for root_dependency in root_dependencies:
            if "optional" in root_dependencies[root_dependency]:
                del root_dependencies[root_dependency]["optional"]

        # get all the test dependencies from the ragstack-e2e-tests module
        test_dependencies = e2e_tests_data["tool"]["poetry"]["group"]["test"]["dependencies"]

        # merge the RAGstack dependencies into the test dependencies
        merged_dependencies = deep_update(test_dependencies, root_dependencies)

        # save the merged test dependencies back into the test pyproject data
        e2e_tests_data["tool"]["poetry"]["group"]["test"]["dependencies"] = merged_dependencies

        # merge the update data into the test pyproject data
        merged_data = deep_update(e2e_tests_data, update_data)

        # output the combined pyproject file
        with open(output_file_path, 'w') as output_file:
            toml.dump(merged_data, output_file, encoder=toml.TomlPreserveInlineDictEncoder())

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python merge_toml.py pyproject.toml ragstack-e2e-tests/pyproject.toml ragstack-e2e-tests/pyproject.update.toml pyproject.output.toml")
        sys.exit(1)

    rag_stack_file_path = sys.argv[1]
    e2e_tests_file_path = sys.argv[2]
    update_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    generate_pyproject(rag_stack_file_path, e2e_tests_file_path, update_file_path, output_file_path)
    print(f"Generated pyproject: `{output_file_path}` from `{rag_stack_file_path} (tool.poetry.dependencies)` plus `{e2e_tests_file_path}` and `{update_file_path}`")
