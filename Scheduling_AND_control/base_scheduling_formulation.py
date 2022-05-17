from __future__ import division
import pyomo.environ as pe



def bulind_scheduling():
    m = pe.ConcreteModel(name='scheduling_model')
    #SCALARS---------------------
    m.delta=pe.Param(initialize=0.5,doc='lenght of time periods of discretized time grid [units of time]')

    #SETS------------------------
    m.T=pe.RangeSet(0,48,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['M','RL','RS','S','P'],doc='Set of Units')#M: mix, RL: reactor large, RS: reactor small,S: separation, P: Packing
    m.I=pe.Set(initialize=['M','R1','R2','R3','S','P1','P2'], doc='Set of tasks') #'M': Dilution,'R1': Reaction 1 ,'R2': Reaction 2,'R3': Reaction 3,'S': Separation,'P1': Packing 1,'P2': Packing 2
    m.K=pe.Set(initialize=['M1','M2','M3','S1','I1','I2','I3','I4','I5','I6','W1','P1','P2'],doc='Set of states')


    #PARAMETERS------------------
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    #m.eta.display()
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    #m.t_p.display()
    _I_i_k_minus={}
    _I_i_k_minus['M','S1']=1
    _I_i_k_minus['M','M1']=1

    _I_i_k_minus['R1','M2']=1
    _I_i_k_minus['R1','M3']=1

    _I_i_k_minus['R2','I1']=1
    _I_i_k_minus['R2','I2']=1

    _I_i_k_minus['R3','I3']=1
    _I_i_k_minus['R3','M3']=1

    _I_i_k_minus['S','I4']=1

    _I_i_k_minus['P1','I5']=1

    _I_i_k_minus['P2','I6']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['M','I1']=1

    _I_i_k_plus['R1','I2']=1

    _I_i_k_plus['R2','I3']=1
    _I_i_k_plus['R2','I5']=1

    _I_i_k_plus['R3','I4']=1

    _I_i_k_plus['S','W1']=1
    _I_i_k_plus['S','I6']=1

    _I_i_k_plus['P1','P1']=1

    _I_i_k_plus['P2','P2']=1
    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")


    _rho_minus={}
    _rho_minus['M','S1']=3/5
    _rho_minus['M','M1']=2/5

    _rho_minus['R1','M2']=1/2
    _rho_minus['R1','M3']=1/2

    _rho_minus['R2','I1']=1/2
    _rho_minus['R2','I2']=1/2

    _rho_minus['R3','I3']=1/2
    _rho_minus['R3','M3']=1/2

    _rho_minus['S','I4']=1

    _rho_minus['P1','I5']=1

    _rho_minus['P2','I6']=1   
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")


    _rho_plus={}
    _rho_plus['M','I1']=1

    _rho_plus['R1','I2']=1

    _rho_plus['R2','I3']=3/5
    _rho_plus['R2','I5']=2/5

    _rho_plus['R3','I4']=1

    _rho_plus['S','W1']=2/5
    _rho_plus['S','I6']=3/5

    _rho_plus['P1','P1']=1

    _rho_plus['P2','P2']=1
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")



    _I_i_j_prod={}
    _I_i_j_prod['M','M']=1

    _I_i_j_prod['R1','RL']=1
    _I_i_j_prod['R1','RS']=1

    _I_i_j_prod['R2','RL']=1
    _I_i_j_prod['R2','RS']=1

    _I_i_j_prod['R3','RL']=1
    _I_i_j_prod['R3','RS']=1

    _I_i_j_prod['S','S']=1

    _I_i_j_prod['P1','P']=1

    _I_i_j_prod['P2','P']=1
    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    m.tau_p=pe.Param













    return m
if __name__ == "__main__":
    m=bulind_scheduling()
    #m.pprint()
