from tradingagents.graph.trading_graph import TradingAgentsGraph

# Example of how to run the TradingAgentsGraph
if __name__ == "__main__":
    # Initialize the graph
    # Set debug=False to run without detailed logging to the console
    ta = TradingAgentsGraph(debug=True) 

    # Propagate the graph for a specific company and date
    # The propagate method returns the final agent state and the decision
    final_state, decision = ta.propagate("NVDA", "2024-05-10")

    # --- FIX FOR HANDLING GEMINI'S LIST OUTPUT ---
    # Some models, like Gemini, might return the final decision as a list of strings.
    # This code checks if 'decision' is a list and joins it into a single string.
    if isinstance(decision, list):
        decision = "\n".join(str(item) for item in decision)

    # This also ensures the report within the final_state is properly formatted as a string
    # for any subsequent processing, like the reflection step.
    if final_state and isinstance(final_state.get("final_trade_decision"), list):
        final_state["final_trade_decision"] = "\n".join(str(item) for item in final_state["final_trade_decision"])
    # --- END OF FIX ---
    
    # Print the final trading decision
    print("--- FINAL DECISION ---")
    print(decision)

    # (Optional) Example of how to run the reflection step after a trade
    # You would replace `0.05` with the actual return/loss of the trade.
    # ta.reflect_and_remember(returns_losses=0.05)