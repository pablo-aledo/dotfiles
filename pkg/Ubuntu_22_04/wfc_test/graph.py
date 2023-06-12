import networkx as nx
import z3 as z3
from enum import Enum
import random
from datetime import datetime

class OperationTypes(Enum):
    OPEN = 1
    BOUNDED = 2
    CLOSE = 3
    UNBOUNDED = 4
    INIT = 5

class Operations(Enum):
    INIT = 0
    OPEN = 1
    CLOSE = 2
    WRITE = 3
    READ = 4
    SEEK = 5
    TRUNCATE = 6
    SYNC = 7
    DELETE = 8
    RENAME = 9
    LINK = 10
    UNLINK = 11
    MKDIR = 12
    RMDIR = 13
    CHDIR = 14
    CHMOD = 15
    CHOWN = 16
    UTIME = 17
    STAT = 18
    LSTAT = 19
    ACCESS = 20

def add_operations_from_types(solver, graph):
    for node in graph.nodes():
        solver.add( z3.Implies( graph.nodes()[node]['operation_type'] == OperationTypes.INIT.value, graph.nodes()[node]['operation'] == Operations.INIT.value ) )
        solver.add( z3.Implies( graph.nodes()[node]['operation_type'] == OperationTypes.OPEN.value, graph.nodes()[node]['operation'] == Operations.OPEN.value ) )
        solver.add( z3.Implies( graph.nodes()[node]['operation_type'] == OperationTypes.CLOSE.value, graph.nodes()[node]['operation'] == Operations.CLOSE.value ) )
        solver.add( z3.Implies( graph.nodes()[node]['operation_type'] == OperationTypes.BOUNDED.value, z3.And(graph.nodes()[node]['operation'] >= Operations.WRITE.value, graph.nodes()[node]['operation'] <= Operations.SYNC.value ) ) )
        solver.add( z3.Implies( graph.nodes()[node]['operation_type'] == OperationTypes.UNBOUNDED.value, z3.And(graph.nodes()[node]['operation'] >= Operations.DELETE.value, graph.nodes()[node]['operation'] <= Operations.ACCESS.value ) ) )

def add_operation_constraints(solver, graph):
    for node in graph.nodes():
        solver.add( graph.nodes()[node]['operation_type'] >= OperationTypes.OPEN.value, graph.nodes()[node]['operation_type'] <= OperationTypes.INIT.value )
        solver.add( graph.nodes()[node]['operation'] >= Operations.INIT.value, graph.nodes()[node]['operation'] <= Operations.ACCESS.value )
        solver.add( graph.nodes()[node]['transaction'] >= -1, graph.nodes()[node]['transaction'] <= 100 )
        solver.add( graph.nodes()[node]['buffer_id'] >= 0, graph.nodes()[node]['buffer_id'] < 10 )
        solver.add( graph.nodes()[node]['file_id'] >= 0, graph.nodes()[node]['file_id'] < 10 )
        solver.add( graph.nodes()[node]['file2_id'] >= 0, graph.nodes()[node]['file2_id'] < 10 )
        solver.add( graph.nodes()[node]['offset'] >= 0, graph.nodes()[node]['offset'] < 10000 )
        solver.add( graph.nodes()[node]['size'] >= 0, graph.nodes()[node]['size'] < 100 )

def add_time_constraints(solver, graph):
    for node in graph.nodes():
        solver.add( graph.nodes()[node]['time'] >= -1, graph.nodes()[node]['time'] <= 10000 )

def add_if_open_then_close(solver, graph):
   for node1 in graph.nodes():
       orcondition = False
       othernodes = set(graph.adj[node1])
       for node2 in othernodes:
           nodep  = node2
           nodesn = othernodes - {node2}
           andcondition = True
           andcondition = z3.And( andcondition, z3.And(graph.nodes()[nodep]['operation_type'] == OperationTypes.CLOSE.value, graph.nodes()[nodep]['transaction'] == graph.nodes()[node1]['transaction']) )
           for node3 in nodesn:
               andcondition = z3.And(andcondition, z3.Or(graph.nodes()[node3]['operation_type'] != OperationTypes.CLOSE.value, z3.And(graph.nodes()[node3]['operation_type'] == OperationTypes.CLOSE.value, graph.nodes()[node3]['transaction'] != graph.nodes()[node1]['transaction']) ) )
           orcondition = z3.Or(orcondition, andcondition)
       solver.add( z3.Implies( graph.nodes()[node1]['operation_type'] == OperationTypes.OPEN.value, orcondition ) )

def add_if_close_then_open(solver, graph):
   for node1 in graph.nodes():
       orcondition = False
       othernodes = set(graph.adj[node1])
       for node2 in othernodes:
           nodep  = node2
           nodesn = othernodes - {node2}
           andcondition = True
           andcondition = z3.And( andcondition, z3.And(graph.nodes()[nodep]['operation_type'] == OperationTypes.OPEN.value, graph.nodes()[nodep]['transaction'] == graph.nodes()[node1]['transaction']) )
           for node3 in nodesn:
               andcondition = z3.And(andcondition, z3.Or(graph.nodes()[node3]['operation_type'] != OperationTypes.OPEN.value, z3.And(graph.nodes()[node3]['operation_type'] == OperationTypes.OPEN.value, graph.nodes()[node3]['transaction'] != graph.nodes()[node1]['transaction']) ) )
           orcondition = z3.Or(orcondition, andcondition)
       solver.add( z3.Implies( graph.nodes()[node1]['operation_type'] == OperationTypes.CLOSE.value, orcondition ) )

def add_if_op_then_open(solver, graph):
    for node1 in graph.nodes():
        orcondition = False
        othernodes = set(graph.adj[node1])
        for node2 in othernodes:
            nodep  = node2
            nodesn = othernodes - {node2}
            andcondition = True
            andcondition = z3.And( andcondition, z3.And(graph.nodes()[nodep]['operation_type'] == OperationTypes.OPEN.value, graph.nodes()[nodep]['transaction'] == graph.nodes()[node1]['transaction']) )
            # for node3 in nodesn:
                # andcondition = z3.And(andcondition, z3.Or(graph.nodes()[node3]['operation_type'] != OperationTypes.OPEN.value, z3.And(graph.nodes()[node3]['operation_type'] == OperationTypes.OPEN.value, graph.nodes()[node3]['transaction'] != graph.nodes()[node1]['transaction']) ) )
            orcondition = z3.Or(orcondition, andcondition)

        solver.add( z3.Implies( graph.nodes()[node1]['operation_type'] == OperationTypes.BOUNDED.value, orcondition ) )

def add_close_after_open(solver, graph):
    for node1 in graph.nodes():
        othernodes = set(graph.adj[node1])
        for node2 in othernodes:
            solver.add( z3.Implies( z3.And( graph.nodes()[node1]['operation_type'] == OperationTypes.OPEN.value, graph.nodes()[node2]['operation_type'] == OperationTypes.CLOSE.value, graph.nodes()[node1]['transaction'] == graph.nodes()[node2]['transaction'] ) , graph.nodes()[node1]['time'] < graph.nodes()[node2]['time'] ) )

def add_open_different_transactions(solver, graph):
    for node1 in graph.nodes():
        othernodes = set(graph.adj[node1])
        for node2 in othernodes:
            solver.add( z3.Implies( z3.And( graph.nodes()[node1]['operation_type'] == OperationTypes.OPEN.value, graph.nodes()[node2]['operation_type'] == OperationTypes.OPEN.value ) , graph.nodes()[node1]['transaction'] != graph.nodes()[node2]['transaction'] ) )

def add_unbounded_no_transaction(solver, graph):
    for node1 in graph.nodes():
        solver.add( z3.Implies( graph.nodes()[node1]['operation_type'] == OperationTypes.UNBOUNDED.value, graph.nodes()[node1]['transaction'] == -1 ) )
        solver.add( z3.Implies( graph.nodes()[node1]['operation_type'] != OperationTypes.UNBOUNDED.value, graph.nodes()[node1]['transaction'] != -1 ) )

def add_init_constraints(solver, graph):
    for node1 in graph.nodes():
        solver.add( z3.Implies( graph.nodes()[node1]['operation_type'] == OperationTypes.INIT.value, graph.nodes()[node1]['time'] == -1 ) )
        solver.add( z3.Implies( graph.nodes()[node1]['operation_type'] != OperationTypes.INIT.value, graph.nodes()[node1]['time'] != -1 ) )

def add_op_after_open(solver, graph):
    for node1 in graph.nodes():
        othernodes = set(graph.adj[node1])
        for node2 in othernodes:
            solver.add( z3.Implies( z3.And( graph.nodes()[node1]['operation_type'] == OperationTypes.OPEN.value, graph.nodes()[node2]['operation_type'] == OperationTypes.BOUNDED.value, graph.nodes()[node1]['transaction'] == graph.nodes()[node2]['transaction']) , graph.nodes()[node1]['time'] < graph.nodes()[node2]['time'] ) )

def add_op_before_close(solver, graph):
    for node1 in graph.nodes():
        othernodes = set(graph.adj[node1])
        for node2 in othernodes:
            solver.add( z3.Implies( z3.And( graph.nodes()[node1]['operation_type'] == OperationTypes.BOUNDED.value, graph.nodes()[node2]['operation_type'] == OperationTypes.CLOSE.value, graph.nodes()[node1]['transaction'] == graph.nodes()[node2]['transaction'] ) , graph.nodes()[node1]['time'] < graph.nodes()[node2]['time'] ) )

def get_open_node(solver, graph, node):
    transaction = solver.model()[ graph.nodes()[node]["transaction"] ].as_long()
    for node in graph.nodes():
        optype = solver.model()[ graph.nodes()[node]["operation_type"] ].as_long()
        open_transaction = solver.model()[ graph.nodes()[node]["transaction"] ].as_long()
        if optype == OperationTypes.OPEN.value and transaction == open_transaction:
            return node

def add_no_write_existing(solver, graph):
    solver.check()
    model = solver.model()
    inits = set()
    for node1 in graph.nodes():
        my_optype = model[ graph.nodes()[node1]["operation_type"] ].as_long()
        if my_optype != OperationTypes.INIT.value:
            continue
        inits.add(node1)

    for node1 in graph.nodes():
        my_op = model[ graph.nodes()[node1]["operation"] ].as_long()
        if my_op != Operations.WRITE.value and my_op != Operations.TRUNCATE.value:
            continue

        my_open = get_open_node(solver, graph, node1)

        for node2 in inits:
            solver.add( graph.nodes()[my_open]['file_id'] != graph.nodes()[node2]['file_id'] )

def add_init_different_files(solver, graph):
    solver.check()
    model = solver.model()
    inits = set()
    for node1 in graph.nodes():
        my_optype = model[ graph.nodes()[node1]["operation_type"] ].as_long()
        if my_optype != OperationTypes.INIT.value:
            continue
        inits.add(node1)

    for node1 in inits:
        for node2 in inits:
            if node1 != node2:
                solver.add( graph.nodes()[node1]['file_id'] != graph.nodes()[node2]['file_id'] )

def add_no_double_fds(solver, graph):
    solver.check()
    model= solver.model()

    for node1 in graph.nodes():
        my_optype = model[ graph.nodes()[node1]["operation_type"] ].as_long()
        if my_optype != OperationTypes.OPEN.value:
            continue

        my_transaction = model[ graph.nodes()[node1]["transaction"] ].as_long()

        othernodes = set(graph.adj[node1])

        open_time = model[ graph.nodes()[node1]["time"] ].as_long()
        close_time = -1
        for node2 in othernodes:
            other_optype = model[ graph.nodes()[node2]["operation_type"] ].as_long()
            other_transaction = model[ graph.nodes()[node2]["transaction"] ].as_long()
            if other_optype == OperationTypes.CLOSE.value and other_transaction == my_transaction:
                close_time = model[ graph.nodes()[node2]["transaction"] ].as_long()
                break


        for node2 in othernodes:
            other_optype = model[ graph.nodes()[node2]["operation_type"] ].as_long()
            if other_optype != OperationTypes.OPEN.value:
                continue

            other_open_time = model[ graph.nodes()[node2]["time"] ].as_long()

            if other_open_time >= open_time and other_open_time <= close_time:
                solver.add( graph.nodes()[node1]['file_id'] != graph.nodes()[node2]['file_id'] )

def get_close_time(solver, graph, node):
    open_transaction = solver.model()[ graph.nodes()[node]["transaction"] ].as_long()
    for node in graph.nodes():
        optype = solver.model()[ graph.nodes()[node]["operation_type"] ].as_long()
        transaction = solver.model()[ graph.nodes()[node]["transaction"] ].as_long()
        if optype == OperationTypes.CLOSE.value and transaction == open_transaction:
            return solver.model()[ graph.nodes()[node]["time"] ].as_long()


def add_no_modify_intransit(solver, graph):
    solver.check()
    model = solver.model()
    for node1 in graph.nodes():
        my_optype = model[ graph.nodes()[node1]["operation_type"] ].as_long()
        my_time = model[ graph.nodes()[node1]["time"] ].as_long()
        if my_optype != OperationTypes.UNBOUNDED.value:
            continue


        othernodes = set(graph.adj[node1])

        for node2 in othernodes:
            other_optype = model[ graph.nodes()[node2]["operation_type"] ].as_long()
            if other_optype != OperationTypes.OPEN.value:
                continue

            other_open_time = model[ graph.nodes()[node2]["time"] ].as_long()
            other_close_time = get_close_time(solver, graph, node2)

            if other_open_time <= my_time and other_close_time >= my_time :
                solver.add( graph.nodes()[node1]['file_id'] != graph.nodes()[node2]['file_id'] )
                solver.add( graph.nodes()[node1]['file2_id'] != graph.nodes()[node2]['file_id'] )

def show_solution(model, graph):
    times = set()
    for node in graph.nodes():
        times.add( model[graph.nodes()[node]['time'] ].as_long() )

    for t in sorted(times):
        for node in graph.nodes():
            if model[ graph.nodes()[node]["time"] ].as_long() == t :
                node = graph.nodes()[node]
                time        = model[ node['time'] ].as_long()
                operation   = model[ node['operation'] ].as_long()
                transaction = model[ node['transaction'] ].as_long()
                buffer_id = model[ node['buffer_id'] ].as_long()
                file_id = model[ node['file_id'] ].as_long()
                file2_id = model[ node['file2_id'] ].as_long()
                offset = model[ node['offset'] ].as_long()
                size = model[ node['size'] ].as_long()
                print(time, operation, transaction, buffer_id, file_id, file2_id, offset, size)

def generate_nodes(n):
    ret = list()
    for x in range(0,n):
        node = {
                "name": "node_%04d" % x,
                "collapsed": False,
                "time": z3.Int("time_%04d" % x),
                "operation_type": z3.Int("operation_type_%04d" % x),
                "operation": z3.Int("operation_%04d" % x),
                "transaction": z3.Int("transaction_%04d" % x),
                "buffer_id": z3.Int("buffer_id_%04d" % x),
                "file_id": z3.Int("file_id_%04d" % x),
                "file2_id": z3.Int("file2_id_%04d" % x),
                "offset": z3.Int("offset_%04d" % x),
                "size": z3.Int("size_%04d" % x)
                }
        ret.append(node)
    return ret

def likelyhood(model, graph, node):
    return 80

def getandcollapse_step1(graph, solver, rand_node):
    solver.check()
    model = solver.model()
    solver.push()

    solver.check()
    andexpr = z3.And(
            graph.nodes()[rand_node]['operation'] == random.randint(0,20),
            graph.nodes()[rand_node]['file_id'] == random.randint(0,9),
            graph.nodes()[rand_node]['file2_id'] == random.randint(0,9)
    )
    solver.add(andexpr)
    solver.check()
    model = solver.model()

    if model[ graph.nodes()[rand_node]["operation_type"] ].as_long() == OperationTypes.INIT.value:
        andexpr = z3.And(andexpr, graph.nodes()[rand_node]['operation'] == Operations.INIT.value )
    if model[ graph.nodes()[rand_node]["operation_type"] ].as_long() == OperationTypes.OPEN.value:
        andexpr = z3.And(andexpr, graph.nodes()[rand_node]['operation'] == Operations.OPEN.value )
    if model[ graph.nodes()[rand_node]["operation_type"] ].as_long() == OperationTypes.CLOSE.value:
        andexpr = z3.And(andexpr, graph.nodes()[rand_node]['operation'] == Operations.CLOSE.value )
    if model[ graph.nodes()[rand_node]["operation_type"] ].as_long() == OperationTypes.BOUNDED.value:
        andexpr = z3.And(andexpr, graph.nodes()[rand_node]['operation'] == random.randint(Operations.WRITE.value, Operations.SYNC.value) )
    if model[ graph.nodes()[rand_node]["operation_type"] ].as_long() == OperationTypes.UNBOUNDED.value:
        andexpr = z3.And(andexpr, graph.nodes()[rand_node]['operation'] == random.randint(Operations.DELETE.value, Operations.ACCESS.value) )

    andexpr = z3.And(
            graph.nodes()[rand_node]['time'] == model[ graph.nodes()[rand_node]["time"] ].as_long(), 
            graph.nodes()[rand_node]['operation_type'] == model[ graph.nodes()[rand_node]["operation_type"] ].as_long(), 
            graph.nodes()[rand_node]['operation'] == model[ graph.nodes()[rand_node]["operation"] ].as_long(), 
            graph.nodes()[rand_node]['transaction'] == model[ graph.nodes()[rand_node]["transaction"] ].as_long(), 
    )

    solver.pop()
    return andexpr

def getandcollapse_step2(graph, solver, rand_node):
    solver.check()
    model = solver.model()

    solver.push()

    solver.check()
    model = solver.model()
    andexpr = z3.And(
            graph.nodes()[rand_node]['buffer_id'] == random.randint(0,9),
            graph.nodes()[rand_node]['offset'] == random.randint(0,10000),
            graph.nodes()[rand_node]['size'] == random.randint(0,99)
    )

    solver.add(andexpr)
    solver.check()
    model = solver.model()

    andexpr = z3.And(
            graph.nodes()[rand_node]['time'] == model[ graph.nodes()[rand_node]["time"] ].as_long(), 
            graph.nodes()[rand_node]['operation_type'] == model[ graph.nodes()[rand_node]["operation_type"] ].as_long(), 
            graph.nodes()[rand_node]['transaction'] == model[ graph.nodes()[rand_node]["transaction"] ].as_long(), 
            graph.nodes()[rand_node]['operation'] == model[ graph.nodes()[rand_node]["operation"] ].as_long(), 
            graph.nodes()[rand_node]['buffer_id'] == model[ graph.nodes()[rand_node]["buffer_id"] ].as_long(),
            graph.nodes()[rand_node]['file_id'] == model[ graph.nodes()[rand_node]["file_id"] ].as_long(),
            graph.nodes()[rand_node]['file2_id'] == model[ graph.nodes()[rand_node]["file2_id"] ].as_long(),
            graph.nodes()[rand_node]['offset'] == model[ graph.nodes()[rand_node]["offset"] ].as_long(),
            graph.nodes()[rand_node]['size'] == model[ graph.nodes()[rand_node]["size"] ].as_long()
    )

    solver.pop()
    return andexpr

def wave_function_collapse_step1(graph, solver):

    for node in graph.nodes():
        graph.nodes()[node]['collapsed'] = False

    collapsed_nodes = 0
    while collapsed_nodes < 10:
        rand_node = random.randint(1, len(graph.nodes()))
        if graph.nodes()[rand_node]['collapsed']:
            continue

        likely = False
        while not likely:
            andexpr = getandcollapse_step1(graph, solver, rand_node)

            solver.push()
            solver.add( andexpr )
            solver.check()

            likely = random.randint(0, 100) < likelyhood(solver.model(), graph, rand_node)
            if solver.check() != z3.sat:
                likely = false

            if likely:
                graph.nodes()[rand_node]['collapsed'] = True
                collapsed_nodes = collapsed_nodes + 1
            else:
                solver.pop()

def wave_function_collapse_step2(graph, solver):

    for node in graph.nodes():
        graph.nodes()[node]['collapsed'] = False

    collapsed_nodes = 0
    while collapsed_nodes < 10:
        rand_node = random.randint(1, len(graph.nodes()))
        if graph.nodes()[rand_node]['collapsed']:
            continue

        likely = False
        while not likely:
            andexpr = getandcollapse_step2(graph, solver, rand_node)

            solver.push()
            solver.add( andexpr )
            solver.check()

            likely = random.randint(0, 100) < likelyhood(solver.model(), graph, rand_node)
            if solver.check() != z3.sat:
                likely = false

            if likely:
                graph.nodes()[rand_node]['collapsed'] = True
                collapsed_nodes = collapsed_nodes + 1
            else:
                solver.pop()

def collapse_optype_time_and_transaction(graph, solver):
    solver.check()
    model = solver.model()
    for node in graph.nodes():
        solver.add( graph.nodes()[node]['operation_type'] == model[ graph.nodes()[node]["operation_type"] ].as_long() )
        solver.add( graph.nodes()[node]['operation'] == model[ graph.nodes()[node]["operation"] ].as_long() )
        solver.add( graph.nodes()[node]['time'] == model[ graph.nodes()[node]["time"] ].as_long() )
        solver.add( graph.nodes()[node]['transaction'] == model[ graph.nodes()[node]["transaction"] ].as_long() )

random.seed(datetime.now().timestamp())

# create an empty undirected graph
G = nx.Graph()

num_nodes = 15

# define the nodes
nodes = generate_nodes(num_nodes)

for n in range(0, num_nodes):
    G.add_nodes_from([(n+1, nodes[n])])

# add the edges to the graph
for i in range(0, num_nodes):
    for j in range(i+1, num_nodes):
        G.add_edges_from([(i+1,j+1)])

# Create a solver
solver = z3.Solver()

#solver.add( G.nodes()[1]['operation'] == Operations.TRUNCATE.value )
#solver.add( G.nodes()[2]['operation'] == Operations.OPEN.value )
#solver.add( G.nodes()[3]['operation'] == Operations.CLOSE.value )
#solver.add( G.nodes()[4]['operation'] == Operations.INIT.value )
#solver.add( G.nodes()[4]['offset'] == 100 )

add_operations_from_types(solver, G)
add_operation_constraints(solver, G)
add_time_constraints(solver, G)
add_if_open_then_close(solver, G)
add_if_close_then_open(solver, G)
add_if_op_then_open(solver, G)
add_close_after_open(solver, G)
add_op_after_open(solver, G)
add_op_before_close(solver, G)
add_open_different_transactions(solver, G)
add_init_constraints(solver, G)
add_unbounded_no_transaction(solver, G)

wave_function_collapse_step1(G, solver)
collapse_optype_time_and_transaction(G, solver)

add_no_write_existing(solver, G)
add_init_different_files(solver, G)
add_no_double_fds(solver, G)
add_no_modify_intransit(solver, G)

wave_function_collapse_step2(G, solver)

# Check if the solver is satisfiable
if solver.check() == z3.sat:
    # Get the model from the solver
    model = solver.model()

    show_solution(model, G)
else:
    # No solution exists that satisfies the constraints
    print("No solution exists")
