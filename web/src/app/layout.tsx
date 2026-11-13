import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rift Oracle — Prédiction de victoire LoL",
  description: "Composez les deux équipes et estimez leurs chances de victoire avec le modèle MlLOL.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
