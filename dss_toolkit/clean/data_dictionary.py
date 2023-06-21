def parse_data_dict_rules(
    data_dict_df,
    parameter_columns,
    orient="grouped",
    undefined_str="<<undefined_rule>>",
):
    """
    Reads the data dictionary dataframe and groups columns based on the similar rules/parameter values

    Returns:
    List of dictionary
        keys: specified in parameter_columns and `columns_affected` ()

    """

    if "valid_values" in parameter_columns:
        parameter_columns = ["data_type"] + parameter_columns  # Add data_type to group for casting valid values

    valid_data_rules_df = data_dict_df[["column_name"] + parameter_columns].copy()
    valid_data_rules_df.fillna(undefined_str, inplace=True)  # Add invalid str

    # Aggregate columns with similar parameters
    agg_rules = (
        valid_data_rules_df.groupby(parameter_columns).agg(columns_affected=("column_name", list)).reset_index()
    )

    # Convert rules into list
    agg_rules_list = agg_rules.to_dict(orient="records")

    # Remove empty parameters
    for rule in agg_rules_list:
        for parameter in parameter_columns:
            # Remove rule parameter if parameter is undefined
            if rule[parameter] == undefined_str:
                rule.pop(parameter)
                continue

            if parameter == "valid_values":
                # Clean comma separated values
                rule["valid_values"] = rule["valid_values"].split(",")

                if rule["data_type"] in ("str", "string"):
                    rule["valid_values"] = [val.strip() for val in rule["valid_values"]]

                if rule["data_type"] in ("int", "integer", "int32"):
                    rule["valid_values"] = [int(val.strip()) for val in rule["valid_values"]]

                if rule["data_type"] in ("float", "float32", "float64"):
                    rule["valid_values"] = [float(val.strip()) for val in rule["valid_values"]]

    # Remove rules with zero parameters (only "columns_affected","data_type" left)
    agg_rules_list_ret = []
    for rule in agg_rules_list:
        if "valid_values" in parameter_columns:
            rule.pop("data_type")
        if len(rule.keys()) > 1:
            agg_rules_list_ret.append(rule)

    if orient == "grouped":
        return agg_rules_list_ret
    else:
        return_list = []
        for grouped_rule in agg_rules_list_ret:
            for column in grouped_rule["columns_affected"]:
                d = dict()
                d["column"] = column
                grouped_rule_copy = dict(grouped_rule)
                grouped_rule_copy.pop("columns_affected")
                d.update(grouped_rule_copy)
                return_list.append(d)
        return return_list
