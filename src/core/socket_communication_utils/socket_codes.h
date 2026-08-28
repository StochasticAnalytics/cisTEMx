#ifndef __SOCKET_CODES_H__
#define __SOCKET_CODES_H__

#define SOCKET_CODE_SIZE 16

#define SETUP_SOCKET_CODES unsigned char socket_input_buffer[SOCKET_CODE_SIZE];
const unsigned char socket_please_identify[]             = "JcFG>&P.RuC9,>za";
const unsigned char socket_sending_identification[]      = "gC2CeZWNb2GPv5qh";
const unsigned char socket_you_are_connected[]           = "J82zjSwYY^-!bF>4";
const unsigned char socket_send_job_details[]            = "gr<V>ThBp6w9fzLg";
const unsigned char socket_sending_job_package[]         = "'8ujA!Lup%PR*!hG";
const unsigned char socket_you_are_the_master[]          = "eVmYc.3!g}}cZZsz";
const unsigned char socket_you_are_a_worker[]            = "U6u*:z6}W+7nV2g'";
const unsigned char socket_send_next_job[]               = "z7PnJh=x;[b#f/6L";
const unsigned char socket_time_to_die[]                 = ")[czL7$#Sg/d4-*K";
const unsigned char socket_ready_to_send_single_job[]    = "-TDv(X*kY.:d`D5:";
const unsigned char socket_i_have_an_error[]             = "8TU.cDc3jr,rb[SN";
const unsigned char socket_i_have_info[]                 = "+5nxvY@zt.!_R#Vn";
const unsigned char socket_job_finished[]                = "jNA[3!VdLdkb$LwM";
const unsigned char socket_number_of_connections[]       = "Uu6tsQ,z}M''T`7f";
const unsigned char socket_i_am_a_dedicated_master[]     = "Dm8@ku!W3q#zLc%R";
const unsigned char socket_all_jobs_finished[]           = "aL)yaH[$3s;9Ymk6";
const unsigned char socket_job_result[]                  = "3F6E_.``L6YC^q[U";
const unsigned char socket_job_result_queue[]            = "^}`@pF9m;{m9k=$F";
const unsigned char socket_result_with_image_to_write[]  = "=z4-Y8Ge?vEjh`H^";
const unsigned char socket_program_defined_result[]      = "e}w<S9hm<3L6Dr+V";
const unsigned char socket_send_thread_timing[]          = "Kq04etrq1fO2QV4d";
const unsigned char socket_template_match_result_ready[] = "EP927e$*cQ^egWq'";
// Liveness probe pair (2026-08-27): the master pings sockets holding a job; the peer's
// main thread answers with a pong. No payload follows either code. Both sides of a run
// ship in the same image, so there is no old/new mixing to worry about; an old binary
// that somehow received one would hit the unknown-code branch and drop the socket, which
// for a liveness check is the correct outcome anyway.
const unsigned char socket_liveness_ping[] = "Lv?9V+wQ2@xR-e5N";
const unsigned char socket_liveness_pong[] = "qN4&yZ8_uT!fB^0c";
// Worker -> master (2026-08-28): "a signal is taking me down". Payload: one int, the
// signal number the worker caught. The cisTEM condor submit files set kill_sig = SIGUSR1
// (and must NOT set remove_kill_sig: condor_starter's OsProc::ShutdownGraceful uses the
// remove signal for EVERY graceful shutdown when it is defined), so anything condor takes
// down gracefully - a vacate, a hold, a condor_rm - arrives as SIGUSR1, while a hand kill
// or a wrapper timeout arrives as SIGTERM. The master charges the lost job to the eviction
// budget or the failure budget accordingly (MyApp::HandleWorkerLoss). Sent from the worker's MAIN thread by its
// signal-poll timer; the worker then waits for the master to close the socket.
const unsigned char socket_worker_vacated[] = "Vc!7pQ2m#Rw9-Ue3";

#endif
