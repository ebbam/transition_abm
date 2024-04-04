

#sim_record_t_all = run_sim(True, mod_data, T, 1)
#request a cluster
cluster = ipp.Cluster(n = 4)
cluster.start_cluster_sync()
rc = cluster.connect_client_sync()
rc.wait_for_engines()
direct = rc[:] # use all engines
direct.block=True
direct.run("imports.py")
direct.run("init.py")
direct.run("funs.py")
direct.run("toy_init.py")
#direct.run("us_init.py")
asyncresult = direct.apply_async(run_sim, True, mod_data, T, 1)
#     #asyncresult = view.map_async(lambda x, y: x + y, range(10), range(10))
#     # wait interactively for results
asyncresult.wait_interactive()
#     # retrieve actual results
result = asyncresult.get()
result  
