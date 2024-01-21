import numpy as np
import pandas as pd

class Generator:
    def __init__(self, asset_names):
        self.asset_names = asset_names
        self.data = pd.DataFrame()

    def simulate_brownian_motion(self, volatilities, dt, num_steps, correlation_matrix):       
        num_assets = len(correlation_matrix)
        
        # Generate uncorrelated Brownian motions (should vol be squared ?? )
        uncorrelated_bms = volatilities * dt * np.random.normal(0, 1, size=(num_steps, num_assets))

        # Cholesky decomposition of the correlation matrix
        # https://en.wikipedia.org/wiki/Cholesky_decomposition
        L = np.linalg.cholesky(correlation_matrix)

        # Multiply uncorrelated Brownian motions by Cholesky matrix to get correlated Brownian motions
        return uncorrelated_bms @ L.T
    
    def simulate_rates(self, initial_levels, volatilities, dt, num_steps, correlation_matrix, lognormal=False):       
        bm = self.simulate_brownian_motion(volatilities, dt, (num_steps-1), correlation_matrix)
        
        if lognormal:
            prices = initial_levels * np.exp(np.cumsum(bm, axis=0))
            # add inital stage
            prices = np.concatenate([[initial_levels], prices], axis = 0)
        else:
            bm = np.concatenate([[initial_levels], bm], axis = 0)
            prices =  np.cumsum(bm, axis=0)

        # Calculate cumulative sum to get the Brownian motions        
        time = np.arange(0, num_steps * dt, dt)

        data = pd.DataFrame(prices, columns=self.asset_names)
        data['time'] = time
        data = data.set_index('time')
        self.data = data

        return self.data

if __name__ == "__main__":
    # Parameters
    dt = 0.01  # Time step
    num_steps = 1000  # Number of steps

    # Correlation matrix
    correlation_matrix = np.array([[1.0, 0.5, 0.5],
                                [0.5, 1.0, 0.5],
                                [0.5, 0.5, 1.0]])

    A = Generator(["EURUSD", "EURGBP", "EURSEK"])
    initial_levels =  [1.0, 1.0, 1.0]
    vol = 0.001


    data = A.simulate_rates(initial_levels, vol, dt, num_steps, correlation_matrix)

    # data.diff().plot()

    data.plot()
    plt.show()