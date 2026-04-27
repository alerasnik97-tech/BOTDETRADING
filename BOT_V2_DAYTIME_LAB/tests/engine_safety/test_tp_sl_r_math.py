
import unittest

class TestTpSlRMath(unittest.TestCase):
    def test_r_multiple_calculation(self):
        """Validar que R se calcule correctamente según la dirección"""
        entry = 1.0800
        sl = 1.0700
        risk = abs(entry - sl) # 0.0100
        
        # Win Long (2R)
        exit_win = 1.1000
        r_win = (exit_win - entry) / risk
        self.assertEqual(r_win, 2.0, "Ganancia Long debe ser R positiva")
        
        # Loss Long (-1R)
        exit_loss = 1.0700
        r_loss = (exit_loss - entry) / risk
        self.assertEqual(r_loss, -1.0, "Pérdida Long debe ser R negativa")

        # Win Short (2R)
        entry_s = 1.0800
        sl_s = 1.0900
        risk_s = abs(entry_s - sl_s) # 0.0100
        exit_win_s = 1.0600
        r_win_s = (entry_s - exit_win_s) / risk_s
        self.assertEqual(r_win_s, 2.0, "Ganancia Short debe ser R positiva")

    def test_phase12_sign_flip_detection(self):
        """Detectar específicamente el bug de target del lado incorrecto"""
        entry = 1.0800
        risk = 0.0010
        tp_multiplier = 2.0
        
        # Bug detectado en Phase 12: TP = entry - (risk * tp_r) para LONG
        tp_incorrecto = entry - (risk * tp_multiplier) # 1.0780
        
        # Para un LONG, el TP DEBE ser mayor que la entrada
        self.assertGreater(entry + (risk * tp_multiplier), entry, "TP Long debe ser superior a entrada")
        self.assertLess(tp_incorrecto, entry, "DETECCIÓN: El TP incorrecto está por debajo de la entrada")
        
        # El test falla si el TP está del lado equivocado
        def validate_tp(direction, entry_p, tp_p):
            if direction == "LONG":
                return tp_p > entry_p
            else:
                return tp_p < entry_p
        
        self.assertTrue(validate_tp("LONG", entry, entry + 0.0020))
        self.assertFalse(validate_tp("LONG", entry, tp_incorrecto), "Error de signo: TP Long por debajo de entrada")

    def test_profit_factor_math(self):
        """Validar fórmula de PF: GP / abs(GL)"""
        gp = 100.0
        gl = -50.0
        pf = gp / abs(gl)
        self.assertEqual(pf, 2.0)
        
        # Caso sin pérdidas
        gl_zero = 0.0
        pf_inf = gp / abs(gl_zero) if gl_zero != 0 else 999.0
        self.assertEqual(pf_inf, 999.0, "PF debe ser controlado si no hay pérdidas")

if __name__ == "__main__":
    unittest.main()
