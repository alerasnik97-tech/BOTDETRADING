from shadow_line_lab.shadow_autopilot import coordinator, state_manager, reporting
import sys

def main():
    coord = coordinator.ShadowCoordinator()
    state = coord.execute_full_pipeline()
    
    state_manager.save_overall_status(state)
    state_manager.update_autopilot_log(state)
    reporting.generate_final_reports(state)
    
    print(f"Shadow Autopilot finalizado con estado: {state['overall_status']}")

if __name__ == "__main__":
    main()
