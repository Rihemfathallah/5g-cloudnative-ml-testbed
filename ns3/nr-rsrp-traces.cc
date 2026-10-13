/*
 * Signal traces for the handover study, computed with ns-3's 3GPP TR 38.901 channel models
 * (ThreeGppUmaPropagationLossModel + ThreeGppUmaChannelConditionModel -- the propagation models
 * that 5G-LENA uses for NR).
 *
 * Input : trajectories CSV (traj,time_s,speed_kmh,x,y) exported by `tb handover export-trajectories`,
 *         so ns-3 and the Python producer use IDENTICAL UE paths.
 * Output: traj,time_s,speed_kmh,x,y,cell,rsrp_dbm   (large-scale RSRP per resource element)
 *
 * Build (inside an ns-3 tree):  cp nr-rsrp-traces.cc scratch/ && ./ns3 build scratch/nr-rsrp-traces
 * Run   : ./ns3 run "scratch/nr-rsrp-traces --input=traj.csv --output=out.csv --seed=1"
 *
 * Parameters mirror configs/handover.yaml (deployment: 7 sites, ISD 500 m, 25 m / 1.5 m,
 * 3.5 GHz, 46 dBm over 51 PRB x 12 subcarriers). Keep them in sync.
 *
 * Status: written against the ns-3.42 API; NOT compiled in this repository's CI.
 */
#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/propagation-module.h"

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

using namespace ns3;

struct Sample { uint32_t traj; double t, speed, x, y; };

int main(int argc, char* argv[])
{
    std::string input = "traj.csv", output = "out.csv";
    uint32_t seed = 1;
    double fcHz = 3.5e9, isd = 500.0, hBs = 25.0, hUt = 1.5, txDbm = 46.0;
    uint32_t nRe = 51 * 12;
    CommandLine cmd(__FILE__);
    cmd.AddValue("input", "trajectory CSV", input);
    cmd.AddValue("output", "RSRP CSV", output);
    cmd.AddValue("seed", "RNG run number", seed);
    cmd.Parse(argc, argv);
    RngSeedManager::SetSeed(12345);
    RngSeedManager::SetRun(seed);

    // ---- read trajectories
    std::ifstream in(input);
    std::string line;
    std::getline(in, line);  // header
    std::map<uint32_t, std::vector<Sample>> traj;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        Sample s; char c;
        ss >> s.traj >> c >> s.t >> c >> s.speed >> c >> s.x >> c >> s.y;
        traj[s.traj].push_back(s);
    }

    // ---- 7 hexagonal sites (centre + first ring), same order as testbed/handover/channel.py
    std::vector<Vector> sites{Vector(0, 0, hBs)};
    for (int k = 0; k < 6; ++k) {
        double a = M_PI / 6 + k * M_PI / 3;
        sites.emplace_back(isd * std::cos(a), isd * std::sin(a), hBs);
    }
    double txPerRe = txDbm - 10 * std::log10(nRe);

    std::ofstream out(output);
    out << "traj,time_s,speed_kmh,x,y,cell,rsrp_dbm\n";
    for (auto& [id, samples] : traj) {
        // One loss model per trajectory: its channel-condition and shadowing state follows the UE.
        Ptr<ThreeGppUmaChannelConditionModel> cond = CreateObject<ThreeGppUmaChannelConditionModel>();
        cond->SetAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));   // spatially consistent via mobility
        Ptr<ThreeGppUmaPropagationLossModel> loss = CreateObject<ThreeGppUmaPropagationLossModel>();
        loss->SetAttribute("Frequency", DoubleValue(fcHz));
        loss->SetAttribute("ShadowingEnabled", BooleanValue(true));
        loss->SetChannelConditionModel(cond);
        int64_t stream = 1000 * id;
        stream += cond->AssignStreams(stream);
        loss->AssignStreams(stream);

        std::vector<Ptr<MobilityModel>> bs;
        for (auto& p : sites) {
            auto m = CreateObject<ConstantPositionMobilityModel>();
            m->SetPosition(p);
            bs.push_back(m);
        }
        auto ue = CreateObject<ConstantPositionMobilityModel>();
        for (auto& s : samples) {
            // Advance simulated time so time-based state (condition updates, shadowing) progresses.
            Simulator::Schedule(Seconds(s.t) - Simulator::Now(), [](){});
            Simulator::Run();
            ue->SetPosition(Vector(s.x, s.y, hUt));
            for (size_t c = 0; c < bs.size(); ++c) {
                double rx = loss->CalcRxPower(txPerRe, ue, bs[c]);
                out << s.traj << ',' << s.t << ',' << s.speed << ',' << s.x << ',' << s.y << ','
                    << c << ',' << rx << '\n';
            }
        }
        Simulator::Destroy();
    }
    return 0;
}
