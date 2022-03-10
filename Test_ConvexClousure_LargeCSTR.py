from scipy import optimize
import itertools as it
import math
import numpy as np
import pyomo.environ as pe
from pyomo.opt.base.solvers import SolverFactory
import os
from decimal import Decimal


def neighborhood_k_eq_inf(dimension: int = 2) -> dict:
    """
    Function creates a k=Infinity neighborhood of the given dimension
    Args:
        dimension: Dimension of the neighborhood
    Returns:
        temp: Dictionary contaning in each item a list with a direction within the neighborhood
        TODO change temp name here to something more useful
    """

    neighbors = list(it.product([-1, 0, 1], repeat=dimension))
    directions = {}
    for i in range(len(neighbors)):
        directions[i+1] = list(neighbors[i])
    temp = directions.copy()
    for i in directions.keys():
        if temp[i] == [0]*dimension:
            temp.pop(i, None)
    return temp

def convex_clousure(data,xinitial):
    """
    Data: is a dictionary with tuples for the variables and the objective function value. The convex hull is calculated 
    with respect to this information

    xinitial: Is the point (list) where the convex hull is calculated
    """
    
    A_ub=[]
    b_ub=[]
    x=xinitial.copy()
    x.append(1)
    c=[-i for i in x]
    for i in data:
        
        w=list(i)
        w.append(1)
        A_ub.append(w)
        b_ub.append(data[i])

    minus_solution=optimize.linprog(c=c,A_ub=A_ub,b_ub=b_ub,method='revised simplex',bounds=(None,None))

    #objval=-minus_solution.fun
    #status=minus_solution.message
    variables=minus_solution.x
    return variables

def solve_subproblem_and_neighborhood(x,neigh):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
    Returns:
        generated_dict: A dictionary with the points evaluated and their objective function value (central point and neighborhood).
        generated_list_feasible: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (only feasible).
        generated_list_all: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (both feasible and infeasible).
    """
    infinity_val=1e+9
    Adjustable_val=0.5
    generated_dict={}
    generated_list_feasible=[] #if required
    Internaldata={(1,1): 9.89473684210508,
(2,1): 4.06188168425035,
(2,2): 4.06188168453086,
(3,1): 3.31484540307234,
(3,2): 3.31484186064413,
(3,3): 3.31484282218572,
(4,1): 3.13431061006713,
(4,2): 3.13433225500005,
(4,3): 3.13431721917598,
(4,4): 3.1337803770053,
(5,1): 3.13019813302488,
(5,2): 3.13019813302546,
(5,3): 3.13019800592723,
(5,4): 3.13019442788084,
(5,5): 3.06201284860286,
(6,1): 3.20096671488823,
(6,2): 3.20096350647452,
(6,3): 3.20097374668157,
(6,4): 3.20097659458549,
(6,5): 3.20097493382481,
(6,6): 3.00727797400091,
(7,1): 3.28712577152341,
(7,2): 3.28713478956283,
(7,3): 3.28714013135555,
(7,4): 3.28715135629112,
(7,5): 3.28714020229958,
(7,6): 3.26943383135719,
(7,7): 2.96604707228901,
(8,1): 3.35616678047614,
(8,2): 3.35617132517943,
(8,3): 3.35616953018637,
(8,4): 3.35617339479444,
(8,5): 3.35617521031046,
(8,6): 3.35617523484431,
(8,7): 3.26278659414332,
(8,8): 2.93440510304818,
(9,1): 3.4079286722128,
(9,2): 3.40792118782831,
(9,3): 3.40792545553629,
(9,4): 3.4079286722128,
(9,5): 3.40792867221289,
(9,6): 3.40792867208205,
(9,7): 3.40056895451471,
(9,8): 3.22203656690064,
(9,9): 2.90953050735785,
(10,1): 3.44794215166175,
(10,2): 3.44795149294473,
(10,3): 3.44796403165825,
(10,4): 3.44795301018499,
(10,5): 3.44795115360049,
(10,6): 3.44796089888402,
(10,7): 3.44796402923199,
(10,8): 3.3998301577775,
(10,9): 3.17699509306359,
(10,10): 2.88953600290778,
(11,1): 3.47989707548166,
(11,2): 3.47989432807918,
(11,3): 3.47989829423447,
(11,4): 3.47988393927361,
(11,5): 3.47990034597013,
(11,6): 3.47986076175749,
(11,7): 3.47990167689411,
(11,8): 3.47748200680056,
(11,9): 3.37158458807278,
(11,10): 3.13592546602378,
(11,11): 2.87313982640928,
(12,1): 3.5060040431764,
(12,2): 3.50599788729133,
(12,3): 3.50598828401517,
(12,4): 3.5060040431764,
(12,5): 3.5060040431764,
(12,6): 3.5060040431764,
(12,7): 3.5060040431764,
(12,8): 3.50600242011479,
(12,9): 3.48083709692459,
(12,10): 3.33263967359331,
(12,11): 3.10014973408919,
(12,12): 2.85946151708074,
(13,1): 3.52761911064246,
(13,2): 3.52774968177939,
(13,3): 3.52775382531757,
(13,4): 3.52775383166333,
(13,5): 3.52775383166341,
(13,6): 3.52775383166341,
(13,7): 3.52774311912344,
(13,8): 3.52775383166341,
(13,9): 3.52729153275355,
(13,10): 3.46310066666426,
(13,11): 3.2918721716423,
(13,12): 3.06929394638481,
(13,13): 2.84788780959441,
(14,1): 3.54616199005166,
(14,2): 3.54616316210001,
(14,3): 3.54616552971515,
(14,4): 3.54616537791568,
(14,5): 3.54616169813068,
(14,6): 3.54616561273096,
(14,7): 3.54615883548855,
(14,8): 3.54616561260835,
(14,9): 3.54616527537198,
(14,10): 3.53306232457437,
(14,11): 3.43362695983803,
(14,12): 3.25304984666244,
(14,13): 3.0426414262361,
(14,14): 2.83797911330914,
(15,1): 3.56195863312256,
(15,2): 3.56195863312261,
(15,3): 3.56195863312255,
(15,4): 3.56195863312261,
(15,5): 3.56195609609358,
(15,6): 3.56195862065693,
(15,7): 3.56195863312261,
(15,8): 3.56195858391043,
(15,9): 3.56195863290008,
(15,10): 3.56195764247087,
(15,11): 3.52258589928687,
(15,12): 3.39904459631988,
(15,13): 3.21745094466477,
(15,14): 3.01949128569885,
(15,15): 2.82938995260134,
(16,1): 3.57565784215859,
(16,2): 3.57565255303584,
(16,3): 3.57565784267789,
(16,4): 3.57564622205079,
(16,5): 3.57565784267806,
(16,6): 3.57565784267805,
(16,7): 3.57565595058529,
(16,8): 3.57565784267804,
(16,9): 3.57565784258039,
(16,10): 3.57564771340901,
(16,11): 3.56908090386205,
(16,12): 3.50129429152172,
(16,13): 3.36331547653971,
(16,14): 3.18531780083672,
(16,15): 2.99924240304711,
(16,16): 2.82187780784784,
(17,1): 3.58765581318022,
(17,2): 3.58765493709467,
(17,3): 3.58764805451301,
(17,4): 3.5876525488758,
(17,5): 3.58765581318026,
(17,6): 3.58765581318026,
(17,7): 3.58764491298359,
(17,8): 3.58765277177396,
(17,9): 3.58764430811705,
(17,10): 3.58765569831275,
(17,11): 3.58765062921146,
(17,12): 3.56343009753452,
(17,13): 3.47373233737211,
(17,14): 3.3285376316779,
(17,15): 3.1564666027615,
(17,16): 2.981414255036,
(17,17): 2.81526673800131,
(18,1): 3.59822714835687,
(18,2): 3.59825224357741,
(18,3): 3.59823034738683,
(18,4): 3.59823136421654,
(18,5): 3.59825178556337,
(18,6): 3.59824746132597,
(18,7): 3.59825224339082,
(18,8): 3.59825224357744,
(18,9): 3.59825224357744,
(18,10): 3.59825224357744,
(18,11): 3.59825205126083,
(18,12): 3.59520929591063,
(18,13): 3.54836919994052,
(18,14): 3.44318888380226,
(18,15): 3.29569501425863,
(18,16): 3.13058294352719,
(18,17): 2.96562740301786,
(18,18): 2.80938731569139,
(19,1): 3.60767577747462,
(19,2): 3.60768003310149,
(19,3): 3.60768003228313,
(19,4): 3.60768003310149,
(19,5): 3.60767640025418,
(19,6): 3.60767835840809,
(19,7): 3.60766177078043,
(19,8): 3.60768003309728,
(19,9): 3.60765864486736,
(19,10): 3.6076709393839,
(19,11): 3.6076674546403,
(19,12): 3.60767434373583,
(19,13): 3.59279210683569,
(19,14): 3.52698745478607,
(19,15): 3.41180818769577,
(19,16): 3.26516318480746,
(19,17): 3.10731817394023,
(19,18): 2.95154189178112,
(19,19): 2.80412520782032,
(20,1): 3.61610235400687,
(20,2): 3.61612299319138,
(20,3): 3.61612299319119,
(20,4): 3.61610371692448,
(20,5): 3.61612299319121,
(20,6): 3.61612167874755,
(20,7): 3.61611130806879,
(20,8): 3.61612299319136,
(20,9): 3.61611608687872,
(20,10): 3.61612299319136,
(20,11): 3.61612299319136,
(20,12): 3.61612167598246,
(20,13): 3.61493608322167,
(20,14): 3.5823294417413,
(20,15): 3.501778113802,
(20,16): 3.38088522792846,
(20,17): 3.23700521112013,
(20,18): 3.0863476266162,
(20,19): 2.93890355057882,
(20,20): 2.79940566011486,
(21,1): 3.62372818273103,
(21,2): 3.62372818273043,
(21,3): 3.62371471108974,
(21,4): 3.62371927436209,
(21,5): 3.62371715607057,
(21,6): 3.62372818273107,
(21,7): 3.62372818090554,
(21,8): 3.62372029876258,
(21,9): 3.62372570309115,
(21,10): 3.62372818273107,
(21,11): 3.62372502261477,
(21,12): 3.62372438250039,
(21,13): 3.6236913425305,
(21,14): 3.6147097778424,
(21,15): 3.56593865639503,
(21,16): 3.47461512655594,
(21,17): 3.35114765719976,
(21,18): 3.21112395109829,
(21,19): 3.06737732616683,
(21,20): 2.92751753860168,
(21,21): 2.79512698988929,
(22,1): 3.63061468724473,
(22,2): 3.63061461097205,
(22,3): 3.63060419355643,
(22,4): 3.63060590365433,
(22,5): 3.63061468724509,
(22,6): 3.63059587822376,
(22,7): 3.63061468724508,
(22,8): 3.63061468724509,
(22,9): 3.63059528976525,
(22,10): 3.63061452242401,
(22,11): 3.63061459826846,
(22,12): 3.63060244633523,
(22,13): 3.63061468590678,
(22,14): 3.63030507228624,
(22,15): 3.60761549212852,
(22,16): 3.54548666763288,
(22,17): 3.44678604270338,
(22,18): 3.3229564205445,
(22,19): 3.18737231645102,
(22,20): 3.05016449846549,
(22,21): 2.9171912814745,
(22,22): 2.79124997128243,
(23,1): 3.63687998574753,
(23,2): 3.63685924237674,
(23,3): 3.63687998776756,
(23,4): 3.63685592540085,
(23,5): 3.63687995053206,
(23,6): 3.63686406943401,
(23,7): 3.63687476401165,
(23,8): 3.63687998776772,
(23,9): 3.63687998776772,
(23,10): 3.63687080582476,
(23,11): 3.63686208007727,
(23,12): 3.63686763745523,
(23,13): 3.63687998750468,
(23,14): 3.63687982766666,
(23,15): 3.63158323421935,
(23,16): 3.59515162362513,
(23,17): 3.52248185453509,
(23,18): 3.41918562167549,
(23,19): 3.29645018010503,
(23,20): 3.1655555003451,
(23,21): 3.03448274646178,
(23,22): 2.90780668868242,
(23,23): 2.78770169831211,
(24,1): 3.64260466024987,
(24,2): 3.64260466024416,
(24,3): 3.64260466024989,
(24,4): 3.64259348321509,
(24,5): 3.64260466024862,
(24,6): 3.64258970371646,
(24,7): 3.6425926531518,
(24,8): 3.64259152061295,
(24,9): 3.64260466025002,
(24,10): 3.64260466025004,
(24,11): 3.64259825783302,
(24,12): 3.64260466025004,
(24,13): 3.64260269329812,
(24,14): 3.6426046524722,
(24,15): 3.64258265278048,
(24,16): 3.62697665049498,
(24,17): 3.57869850367792,
(24,18): 3.49810939012178,
(24,19): 3.39209180311821,
(24,20): 3.27166358297391,
(24,21): 3.14548734333082,
(24,22): 3.02014607643675,
(24,23): 2.89922521707879,
(24,24): 2.78446461617861,
(25,1): 3.64782586225702,
(25,2): 3.64785589620899,
(25,3): 3.64785094954164,
(25,4): 3.64785589620916,
(25,5): 3.64785588045657,
(25,6): 3.64784384442446,
(25,7): 3.64785589620914,
(25,8): 3.64782905900996,
(25,9): 3.64785589620914,
(25,10): 3.64784009209984,
(25,11): 3.64785589620916,
(25,12): 3.6478558957646,
(25,13): 3.64784955767459,
(25,14): 3.64785220584871,
(25,15): 3.64785589620915,
(25,16): 3.6449054721,
(25,17): 3.61756299840645,
(25,18): 3.55944625169034,
(25,19): 3.47320225075784,
(25,20): 3.36648655096405,
(25,21): 3.24853761075846,
(25,22): 3.12701247926793,
(25,23): 3.0069852464625,
(25,24): 2.89134720162591,
(25,25): 2.78147842647898
}

    if tuple(x) in Internaldata:
        #assign value to x if feasible
        generated_dict[tuple(x)]=Internaldata[tuple(x)]
        generated_list_feasible=generated_list_feasible+[x] #if required
        #assign value to neighbors
        for j in neigh:
            if tuple(np.array(x)+np.array(neigh[j])) in Internaldata:
                
                generated_dict[tuple(np.array(x)+np.array(neigh[j]))]=Internaldata[tuple(np.array(x)+np.array(neigh[j]))]
                generated_list_feasible=generated_list_feasible+[list((np.array(x)+Adjustable_val*np.array(neigh[j])))]
            else:
                generated_dict[tuple(np.array(x)+np.array(neigh[j]))]=infinity_val #THIS LAINE ACTUALLY HELPS A LOT TO FIND LOCAL SOLUTIONS FASTER!!!!!
    else:
        #Value of infinity if infeasible
        generated_dict[tuple(x)]=infinity_val
        generated_list_feasible=generated_list_feasible+[x]


    return generated_dict,generated_list_feasible



def build_master():

    """
    Function that builds the master problem
        
    """
    #Model
    m=pe.ConcreteModel(name='Master_problem')

    #External variables
    m.x1=pe.Var(within=pe.Integers, bounds=(1,25),initialize=1)
    m.x2=pe.Var(within=pe.Integers, bounds=(1,25),initialize=1)

    #Known constraints (assumption!!! I know constraints a priori)
    m.known=pe.Constraint(expr=m.x2-m.x1<=0)

    #Cuts
    m.cuts=pe.ConstraintList()

    #Objective function
    m.zobj=pe.Var()

    def obj_rule(m):
        return m.zobj
    
    m.fobj=pe.Objective(rule=obj_rule,sense=pe.minimize)

    return m

if __name__ == "__main__":
    # # TEST 1: Inequalities from discrete points

    # #Data to compute convex hull
    # data={(1,1):	9.894736842,
    # (2,1):	4.061881685,
    # (3,1):	3.314845403,
    # (4,1):	3.134337036,
    # (5,1):	3.130198133,
    # (2,2):	4.061881685,
    # (3,2):	3.314845403,
    # (4,2):	3.134337036,
    # (5,2):	3.130198133,
    # (3,3):	3.314845403,
    # (4,3):	3.134337036,
    # (5,3):	3.130198133,
    # (4,4):	3.133783898,
    # (5,4):	3.130198133,
    # (5,5):	3.062014577}


    # #Data used to find inequalities (disctrede grid)
    # x1_values=range(1,6)
    # x2_values=range(1,6)
    # grid=list(it.product(x1_values,x2_values))

    # #Initialization of vectors that contain inequailties values
    # fconvex=[]
    # variable_value=[]
    # for i in grid:
    #     solution=convex_clousure(data,list(i))
    #     fconvex.append(solution[0])
    #     variable_value.append(solution[1])



    # #Print inequalities generated
    # for i in range(0,len(grid)):
    #     print('x1*'+str(variable_value[i][0])+'+x2*'+str(variable_value[i][1])+'+'+str(variable_value[i][2])+'<='+'z')

    # TEST 2: calculate the infinity neighborhood (REQUIRED FOR TEST 3)
    #neigh=neighborhood_k_eq_inf(2)
    #print(neigh)


    # # TEST 3: calculate new points to add to convex clousure calculation
    # partial_neigh=solve_subproblem_and_neighborhood([5,5],neigh)
    # print(partial_neigh)



    #TET 4: Build master problem
#    m=build_master()

    initialization=[1,1]
    fobj_actual=1e+9


    #TEST 5: Solve using cuts: first idea
    neigh=neighborhood_k_eq_inf(2)
    D={
        #(25,24): 2.89134720162591
    }

    maxiter=100
    iterations=range(1,maxiter+1)


    x_dict={}  #value of x at each iteration
    fval_dict={}   #objective function value at each iteration
    lower_bound_dict={}    #lower bound for the objective function. (is it)

    for k in iterations:

        #define model
        m=build_master()

        #if first iteration, initialize
        if k==1:
            x_actual=initialization
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,_=solve_subproblem_and_neighborhood(x_actual,neigh)
        fobj_actual=list(new_values.values())[0]
        fval_dict[k]=fobj_actual  
        #Add points to D
        D.update(new_values)
        #print(new_values)

        #Calculate new convex hull and dd cuts to the current model
        for i in D: #calculate cuts only for current discrete variables in D
            cuts=convex_clousure(D,list(i))
            #print(cuts)
            m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        lower_bound_dict[k]=pe.value(m.zobj)
        if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
        
        #if pe.value(m.zobj)==fobj_actual:
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    print('Cuts are the convex hull of every point in D. This is actually similar to a D-SDA without line search')
    print(x_dict)


    neigh=neighborhood_k_eq_inf(2)
    D={
        #(25,24): 2.89134720162591
    }



    only_feasible_bag=[]



    maxiter=100
    iterations=range(1,maxiter+1)


    x_dict={}  #value of x at each iteration

    for k in iterations:

        #define model
        m=build_master()

        #if first iteration, initialize
        if k==1:
            x_actual=initialization
            only_feasible_bag.extend(list(x) for x in D)
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,feasible_n=solve_subproblem_and_neighborhood(x_actual,neigh)
        #print(new_values)
        #print(feasible_n)
        #print(new_values)
        #print(feasible_n)
        #print(all_n)
        only_feasible_bag.extend(x for x in feasible_n if x not in only_feasible_bag)
        #print(only_feasible_bag)
        #print(only_feasible_bag)
        #Add points to D
        D.update(new_values)
        
        #print(new_values)

        #Calculate new convex hull and dd cuts to the current model
        for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
            cuts=convex_clousure(D,i)
            #print(cuts)
            m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    print('Cuts are the convex hull of every point in D, and its corresponding feasible negihborhod (but at a distance of 0.5)')
    print(x_dict)




    #     #TEST 6: Solve using cuts: second idea
    # neigh=neighborhood_k_eq_inf(2)
    # D={}
    # maxiter=10
    # iterations=range(1,maxiter+1)

    # x_dict={}  #value of x at each iteration

    # for k in iterations:

    #     #define model
    #     m=build_master()

    #     #if first iteration, initialize
    #     if k==1:
    #         x_actual=initialization
    #     #update current value of x in the dictionary
    #     x_dict[k]=x_actual

    #     #print(x_actual)
    #     #calculate objective function for current point and its neighborhood (subproblem)
    #     new_values=solve_subproblem_and_neighborhood(x_actual,neigh)

    #     #Add points to D
    #     D.update(new_values)
    #     #print(D)

    #     #Calculate new convex hull and dd cuts to the current model
    #     #for i in D: #calculate cuts only for current discrete variables in D
    #     cuts=convex_clousure(D,list(x_actual))
    #         #print(cuts)
    #     m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
        
    #     #Solve master problem       
    #     SolverFactory('gams', solver='cplex').solve(m, tee=False)

    #     #Stop?
    #     #print([pe.value(m.x1),pe.value(m.x2)])
    #     if [pe.value(m.x1),pe.value(m.x2)]==x_actual:
    #         break
    #     else:
    #         x_actual=[pe.value(m.x1),pe.value(m.x2)]
    # print('Cuts are the convex hull calculated at current point only')
    # print(x_dict)


    #TEST 7: Solve using cuts: third idea
    neigh=neighborhood_k_eq_inf(2)
    D={
        #(25,24): 2.89134720162591
    }

    maxiter=100
    iterations=range(1,maxiter+1)

    x_dict={}  #value of x at each iteration

    for k in iterations:

        #define model
        m=build_master()

        #if first iteration, initialize
        if k==1:
            x_actual=initialization
        #print(x_actual)

        #update current value of x in the dictionary
        x_dict[k]=x_actual
        #calculate objective function for current point and its neighborhood (subproblem)
        new_values,_=solve_subproblem_and_neighborhood(x_actual,neigh)

        #Add points to D
        D.update(new_values)
        #print(D)
        #Calculate new convex hull and dd cuts to the current model            
        for i in x_dict:
            cuts=convex_clousure(D,x_dict[i])
            #print(cuts)
            m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        #print(new_values)
        if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual: 
        #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
        #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
#        if 
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    print('Cuts calculated from the central points evaluated so far.')
    print(x_dict)



    #     #TEST 8: Solve using cuts: four idea
    # neigh=neighborhood_k_eq_inf(2)
    # D={}
    # maxiter=100
    # iterations=range(1,maxiter+1)

    # x_dict={}  #value of x at each iteration

    # for k in iterations:

    #     #define model
    #     m=build_master()

    #     #if first iteration, initialize
    #     if k==1:
    #         x_actual=initialization
    #     #print(x_actual)

    #     #update current value of x in the dictionary
    #     x_dict[k]=x_actual
    #     #calculate objective function for current point and its neighborhood (subproblem)
    #     new_values=solve_subproblem_and_neighborhood(x_actual,neigh)

    #     #Add points to D
    #     D.update(new_values)
    #     #print(D)

    #     #Calculate new convex hull and dd cuts to the current model
    #     #for i in D: #calculate cuts only for current discrete variables in D
    #     if k==1:
    #         cuts=convex_clousure(D,list(x_actual))
    #             #print(cuts)
    #         m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
    #     else:
    #         for i in D: #calculate cuts only for current discrete variables in D
    #             cuts=convex_clousure(D,list(i))
    #             #print(cuts)
    #             m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)

    #     #Solve master problem       
    #     SolverFactory('gams', solver='baron').solve(m, tee=False)

    #     #Stop?
    #     #print([pe.value(m.x1),pe.value(m.x2)])
    #     if [pe.value(m.x1),pe.value(m.x2)]==x_actual:
    #         break
    #     else:
    #         x_actual=[pe.value(m.x1),pe.value(m.x2)]
    # print('At first iteration, only initialization for cuts. Then, every point in D is used')
    # print(x_dict)


    #MEJOR VERSION HASTA AHORA, PERO CON EL ERROR DE QUE NO ESTA COSIDERANDO LA INFORMACION DEL RANDOM SAMPLING EN LA PRIMERA ITERACION
    neigh=neighborhood_k_eq_inf(2)
    D={
        #(25,24): 2.89134720162591
    }



    only_feasible_bag=[]



    maxiter=100
    iterations=range(1,maxiter+1)


    x_dict={}  #value of x at each iteration

    for k in iterations:

        #define model
        m=build_master()

        #if first iteration, initialize
        if k==1:
            x_actual=initialization
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,feasible_n=solve_subproblem_and_neighborhood(x_actual,neigh)
        #print(new_values)
        #print(feasible_n)
        #print(new_values)
        #print(feasible_n)
        #print(all_n)
        only_feasible_bag.extend(x for x in feasible_n if x not in only_feasible_bag)
        #print(only_feasible_bag)
        #print(only_feasible_bag)
        #Add points to D
        D.update(new_values)
        
        #print(new_values)

        #Calculate new convex hull and dd cuts to the current model
        for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
            cuts=convex_clousure(D,i)
            #print(cuts)
            m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    print('Cuts calculated from the central points evaluated so far, and its corresponding feasible negihborhod (but at a distance of 0.5)')
    print(x_dict)






