import { ToastProvider } from "@/components";
import { NavigationGuardProvider } from "@/lib/navigationGuard";
import { AppRouter } from "./routes/AppRouter";

export function App() {
  return (
    <ToastProvider placement="bottom-right">
      <NavigationGuardProvider>
        <AppRouter />
      </NavigationGuardProvider>
    </ToastProvider>
  );
}
