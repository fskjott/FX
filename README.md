<h1>Introduction</h1>

We wish to explore different methods of handling FX spot risk using two actors: a client trader and market maker.
In this demo we randomly generate trades for the client to execute against the market maker. The market maker can hedge exposure against a "world book" - which serves as an interbank market. If the cost of hedging in the interbank market is equal to the fees charged to the client by the market maker, the market maker will not generate any PnL without warehousing risk.
The entire demo will be based on simulated data.

The flow of the demo is as following:

1. Simulate data
2. Setup Client behavior
3. Setup Market Maker behavior
4. Run experiment

We leave cooler things to be implemented in the future -- I am satisfied that plumbing/flow works resonably well and want a break.
