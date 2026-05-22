import pulp
import random

# Pesos dos critérios
pesos = {'alpha': 1.0, 'beta': 0.5, 'gamma': 2.0, 'delta': 3.0}

class SimulacaoMatchmaking:
    def __init__(self):
        self.motoristas = {}
        self.passageiros = {}
        self.distancias = {} # (m_id, p_id): (tempo, dist)
        self.contador_m = 1  # Para gerar IDs únicos (M1, M2...)
        self.contador_p = 1  # Para gerar IDs únicos (P1, P2...)

    def _gerar_nova_rota_tick(self):
        """Gera tempo e distância com correlação realista (velocidade urbana)."""
        dist = round(random.uniform(1.0, 10.0), 1) # Distância entre 1 e 10 km
        # Velocidade entre 20km/h (3 min/km) e 40km/h (1.5 min/km)
        fator_transito = random.uniform(1.5, 3.0) 
        tempo = round(dist * fator_transito, 1)
        return tempo, dist

    def inicializar_sistema(self, qtd_motoristas, qtd_passageiros):
        """Popula o sistema no Tick 1 com valores 100% aleatórios."""
        for _ in range(qtd_motoristas):
            self.adicionar_agente('motorista')
        for _ in range(qtd_passageiros):
            # Passageiros iniciais podem ter de 0 a 1 de espera inicial para dar variedade
            p_id = self.adicionar_agente('passageiro')
            self.passageiros[p_id]['espera'] = random.randint(0, 1) 

    def adicionar_agente(self, tipo):
        """Cria um novo agente e calcula rotas cruzadas com quem já está no sistema."""
        if tipo == 'motorista':
            m_id = f"M{self.contador_m}"
            self.motoristas[m_id] = {'aval': round(random.uniform(4.0, 5.0), 1)}
            self.contador_m += 1
            # Gera rotas para todos os passageiros existentes
            for p in self.passageiros:
                self.distancias[(m_id, p)] = self._gerar_nova_rota_tick()
            return m_id

        elif tipo == 'passageiro':
            p_id = f"P{self.contador_p}"
            self.passageiros[p_id] = {'espera': 0} # Novos passageiros sempre começam no 0
            self.contador_p += 1
            # Gera rotas a partir de todos os motoristas existentes
            for m in self.motoristas:
                self.distancias[(m, p_id)] = self._gerar_nova_rota_tick()
            return p_id

    def eventos_dinamicos_do_tick(self):
        """Sorteia a entrada de novos usuários entre um tick e outro."""
        print("\n🌐 VERIFICANDO NOVAS SOLICITAÇÕES NO APLICATIVO...")
        novos_m, novos_p = 0, 0
        
        # 60% de chance de entrarem de 1 a 2 novos motoristas
        if random.random() < 0.60:
            novos_m = random.randint(1, 2)
            for _ in range(novos_m): self.adicionar_agente('motorista')
            
        # 70% de chance de entrarem de 1 a 3 novos passageiros
        if random.random() < 0.70:
            novos_p = random.randint(1, 3)
            for _ in range(novos_p): self.adicionar_agente('passageiro')
            
        if novos_m == 0 and novos_p == 0:
            print("  Nenhum usuário novo conectou neste tick.")
        else:
            print(f"  Entraram: {novos_m} novo(s) motorista(s) e {novos_p} novo(s) passageiro(s).")

    def flutuar_condicoes(self):
        """Simula mudanças realistas de trânsito e rotas a cada tick."""
        for chave in self.distancias:
            tempo, dist = self.distancias[chave]
            novo_tempo = max(1.0, tempo + random.uniform(-1.5, 2.0))
            nova_dist = max(0.5, dist + random.uniform(-0.2, 0.2))
            self.distancias[chave] = (round(novo_tempo, 1), round(nova_dist, 1))

    def processar_lote(self, tick, tick_final):
        print(f"\n{'='*60}")
        print(f"🔄 INICIANDO TICK {tick} DE {tick_final}")
        print(f"{'='*60}")
        
        # Ocorrem mudanças de trânsito e entrada de agentes a partir do tick 2
        if tick > 1:
            self.flutuar_condicoes()
            print("🚦 O trânsito se moveu (Tempo e Distância flutuaram ligeiramente).")
            self.eventos_dinamicos_do_tick()

        M = list(self.motoristas.keys())
        N = list(self.passageiros.keys())

        if not M or not N:
            print("\n⚠️ Falta de motoristas ou passageiros para cruzar dados.")
            self.atualizar_filas([])
            return

        prob = pulp.LpProblem(f"Batch_{tick}", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", ((i, j) for i in M for j in N), cat='Binary')

        custos = {}
        print("\n📊 Matriz de Custos do Lote (Quanto menor, melhor):")
        for i in M:
            for j in N:
                tempo, dist = self.distancias.get((i, j), (999, 999))
                aval = self.motoristas[i]['aval']
                espera = self.passageiros[j]['espera']
                
                custos[i,j] = (pesos['alpha']*tempo + pesos['beta']*dist 
                               - pesos['gamma']*aval - pesos['delta']*espera)
                
                print(f"  [{i} -> {j}] Custo: {custos[i,j]:>6.2f} | "
                      f"(Tempo: {tempo:>4}m, Dist: {dist:>4}km, Aval: {aval}⭐, Espera: {espera}t)")

        prob += pulp.lpSum(custos[i,j] * x[i,j] for i in M for j in N)

        for i in M: prob += pulp.lpSum(x[i,j] for j in N) <= 1
        for j in N: prob += pulp.lpSum(x[i,j] for i in M) <= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        matches = [(i, j) for i in M for j in N if x[i,j].varValue == 1]
        
        if tick < tick_final:
            print(f"\n🔮 PROJEÇÃO OTIMIZADA DO TICK {tick} (Decisão retida):")
            if matches:
                for m, p in matches:
                    print(f"  🚕 {m} ➔ {p} (Custo estimado: {custos[m,p]:.2f})")
            else:
                print("  Nenhum match possível.")
            
            print("\n⏳ Segurando despacho. Aumentando o tempo de espera de todos os passageiros na fila...")
            self.atualizar_filas([]) # Lista vazia = ninguém foi retirado
                
        else:
            print(f"\n🏁 TICK FINAL ({tick}) ATINGIDO: Efetivando o Despacho Oficial!")
            if matches:
                for m, p in matches:
                    print(f"  ✅ DESPACHADO: {m} vai buscar {p} (Custo final: {custos[m,p]:.2f})")
            else:
                 print("  Nenhuma corrida viável formada.")
            
            self.atualizar_filas(matches) # Agora sim, removemos os alocados

    def atualizar_filas(self, matches_efetivados):
        """Remove os agentes que foram despachados e soma +1 de espera em quem ficou."""
        for m, p in matches_efetivados:
            del self.motoristas[m]
            del self.passageiros[p]
            
        for p in self.passageiros:
            self.passageiros[p]['espera'] += 1

        print("\n✔️ STATUS DA FILA PARA O PRÓXIMO CICLO:")
        print(f"  Motoristas aguardando: {list(self.motoristas.keys())}")
        print(f"  Passageiros na fila: {list(self.passageiros.keys())}")


# ==========================================
# EXECUÇÃO DA SIMULAÇÃO (Sem sementes fixas)
# ==========================================
if __name__ == "__main__":
    sim = SimulacaoMatchmaking()

    # Inicia o sistema com 3 motoristas e 3 passageiros aleatórios
    sim.inicializar_sistema(qtd_motoristas=3, qtd_passageiros=3)

    TOTAL_TICKS = 4
    
    # Roda a simulação pelo tempo determinado
    for t in range(1, TOTAL_TICKS + 1):
        sim.processar_lote(tick=t, tick_final=TOTAL_TICKS)