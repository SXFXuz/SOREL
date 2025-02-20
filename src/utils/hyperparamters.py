# lr for baselines and (lr,lrd) for sorel
HYPERPARAM_LR = {
    "sgd": {
        "yacht": {
            "cvar": 1e-1,
            "extremile": 1e-1,
            "esrm": 1e-1,
        },
        "energy": {
            "cvar": 1e-1,
            "extremile": 1e-1,
            "esrm": 1e-1,
        },
        "concrete": {
            "cvar": 3e-2,
            "extremile": 3e-2,
            "esrm": 1e-1,
        },
        "kin8nm": {
            "cvar": 1e-3,
            "extremile": 1e-3,
            "esrm": 1e-3,
        },
        "power": {
            "cvar": 1e-2,
            "extremile": 1e-2,
            "esrm": 1e-2,
        },
        "law": {
            "cvar": 1e-4,
            "extremile": 3e-4,
            "esrm": 3e-4,
        },
        "acs": {
            "cvar": 1e-2,
            "extremile": 3e-3,
            "esrm": 3e-3,
        },
        "amazon": {
            "cvar": 3e-1,
            "extremile": 1e-1,
            "esrm": None,
        }
    },
    "lsvrg": {
        "yacht": {
            "cvar": 1e-3,
            "extremile": 3e-3,
            "esrm": 3e-3,
        },
        "energy": {
            "cvar": 1e-3,
            "extremile": 1e-2,
            "esrm": 3e-2,
        },
        "concrete": {
            "cvar": 1e-3,
            "extremile": 3e-3,
            "esrm": 3e-3,
        },
        "kin8nm": {
            "cvar": 1e-4,
            "extremile": 1e-4,
            "esrm": 1e-4,
        },
        "power": {
            "cvar": 3e-4,
            "extremile": 3e-4,
            "esrm": 3e-4,
        },
        "law": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        },
        "acs": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        },
        "amazon": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        }
    },
    "prospect": {
        "yacht": {
            "cvar": 3e-3,
            "extremile": 1e-2,
            "esrm": 1e-2,
        },
        "energy": {
            "cvar": 3e-3,
            "extremile": 1e-2,
            "esrm": 3e-2,
        },
        "concrete": {
            "cvar": 1e-3,
            "extremile": 3e-3,
            "esrm": 3e-3,
        },
        "kin8nm": {
            "cvar": 1e-4,
            "extremile": 1e-4,
            "esrm": 1e-4,
        },
        "power": {
            "cvar": 3e-4,
            "extremile": 3e-4,
            "esrm": 3e-4,
        },
        "law": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        },
        "acs": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        },
        "amazon": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        }
    },
    "sorel": {
        "yacht": {
            "cvar": (1e-2,4e-1),
            "extremile": (3e-2,1e-1),
            "esrm": (3e-2,1e-1),
        },
        "energy": {
            "cvar": (3e-2,1),
            "extremile": (3e-2,4e-1),
            "esrm": (3e-2,4e-1),
        },
        "concrete": {
            "cvar": (1e-2,1e-1),
            "extremile": (1e-2,1e-1),
            "esrm": (1e-2,1e-1),
        },
        "kin8nm": {
            "cvar": (3e-4,1),
            "extremile": (1e-4,2),
            "esrm": (1e-4,1),
        },
        "power": {
            "cvar": (3e-3,2),
            "extremile": (3e-3,1),
            "esrm": (3e-3,1),
        },
        "law": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        },
        "acs": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        },
        "amazon": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        }
    },
    "lsvrg_batch": {
        "yacht": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        },
        "energy": {
            "cvar": 3e-2,
            "extremile": 3e-1,
            "esrm": 3e-1,
        },
        "concrete": {
            "cvar": 3e-2,
            "extremile": 3e-2,
            "esrm": 1e-1,
        },
        "kin8nm": {
            "cvar": 1e-3,
            "extremile": 3e-3,
            "esrm": 3e-3,
        },
        "power": {
            "cvar": 1e-2,
            "extremile": 1e-2,
            "esrm": 3e-2,
        },
        "law": {
            "cvar": 1e-4,
            "extremile": 3e-4,
            "esrm": 3e-4,
        },
        "acs": {
            "cvar": 1e-2,
            "extremile": 3e-3,
            "esrm": 1e-2,
        },
        "amazon": {
            "cvar": 3e-1,
            "extremile": 1e-1,
            "esrm": None,
        }
    },
    "prospect_batch": {
        "yacht": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        },
        "energy": {
            "cvar": 3e-2,
            "extremile": 1e-1,
            "esrm": 3e-1,
        },
        "concrete": {
            "cvar": 1e-2,
            "extremile": 3e-2,
            "esrm": 3e-2,
        },
        "kin8nm": {
            "cvar": 1e-3,
            "extremile": 1e-3,
            "esrm": 1e-3,
        },
        "power": {
            "cvar": 1e-2,
            "extremile": 1e-2,
            "esrm": 1e-2,
        },
        "law": {
            "cvar": 1e-4,
            "extremile": 3e-4,
            "esrm": 3e-4,
        },
        "acs": {
            "cvar": 1e-2,
            "extremile": 3e-3,
            "esrm": 3e-2,
        },
        "amazon": {
            "cvar": 3e-1,
            "extremile": 1e-1,
            "esrm": None,
        }
    },
    "sorel_batch": {
        "yacht": {
            "cvar": None,
            "extremile": None,
            "esrm": None,
        },
        "energy": {
            "cvar": (3e-1,2),
            "extremile": (3e-1,2e-1),
            "esrm": (3e-1,4e-1),
        },
        "concrete": {
            "cvar": (1e-1,1),
            "extremile": (3e-1,2e-1),
            "esrm": (1e-1,1e-1),
        },
        "kin8nm": {
            "cvar": (3e-3,4),
            "extremile": (3e-3,10),
            "esrm": (3e-3,10),
        },
        "power": {
            "cvar": (3e-1,1),
            "extremile": (3e-2,1),
            "esrm": (3e-1,1),
        },
        "law": {
            "cvar": (3e-4,1e3),
            "extremile": (1e-3,1e1),
            "esrm": (1e-3,1e1),
        },
        "acs": {
            "cvar": (1e-1,1),
            "extremile": (1e-2,4e-2),
            "esrm": (1e-1,4e-2),
        },
        "amazon": {
            "cvar": (3e-1,2e-1),
            "extremile": (3e-1,1e-2),
            "esrm": None,
        }
    },
}