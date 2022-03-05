# README

## To run the server & client

The server must be executed first, following the given format below:
```bash
python3 COMP3221_FLServer.py <port-server> <sub-client>
# <sub-client> should be 1 for subsampling of 2 random clients, 0 for aggregating all connected clients

# Example
python3 COMP3221_FLServer.py 6000 0
```

The clients can be executed after the server, using the given format below:
```bash
python3 COMP3221_FLClient.py <client-id> <port-client> <opt-method>
# <client-id> is the ID for the client, example: client1
# <port-client> is the client's port number
# <opt-method> should be 1 for minibatch-GD and 0 for GD

# Example
python3 COMP3221_FLClient.py client2 6002 1
```


Example:

```bash
# Start server first
python3 COMP3221_FLServer.py 6000 0

# In different terminals, execute the following respectively:
python3 COMP3221_FLClient.py client1 6001 0
python3 COMP3221_FLClient.py client2 6002 0
python3 COMP3221_FLClient.py client3 6003 0
python3 COMP3221_FLClient.py client4 6004 0
python3 COMP3221_FLClient.py client5 6005 0
```

## Directory Folders

- `./results/raw/` contains all the raw unparsed output from each client, each customisation option category has its own folder of results.
	- For example:
		- `minibatch5-nosub` means minibatch with `BATCH_SIZE` of 5 with no subsampling.
		- `GD-sub` means gradient descent with subsampling.
- `./results/parsed/` contains all the parsed data, in other words, the average for each iteration for each category.
- `./report/graphs` contains the graphs generated from the parsed data using the script `parse_result.ipynb` (located in the project root folder).

## Note

- `parse_result.ipynb` is a script used to parse the output data to generate the graphs and output the average of loss and accuracy across all clients for each iteration.
- The learning rate can be controlled via `LEARNING_RATE` variable in `COMP3221_FLClient.py`.
- The batch size for the clients can be controlled through the `BATCH_SIZE` variable in `COMP3221_FLClient.py`.
- The initial waiting period for the server-side to wait for the incoming client connections can be controlled via the `INITIAL_WAIT` variable in `COMP3221_FLServer.py`.
- We did not get the chance to implement the incoming connection from the client to server after the server's initial waiting period, therefore the clients must be initiated within the `INITIAL_WAIT` period.
- The client will write the its results to the project root directory in the form of `client1_log.txt`.



