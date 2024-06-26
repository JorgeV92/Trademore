open Lwt.Infix
open Trademore.Alpaca


let () = 
  Lwt_main.run (
    Lwt_io.printf "\nRunning test of account